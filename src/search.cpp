/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2020 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>   // For std::memset
#include <iostream>
#include <sstream>

#include "evaluate.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"

namespace Search {

  LimitsType Limits;
}

namespace Tablebases {

  int Cardinality;
  bool RootInTB;
  bool UseRule50;
  Depth ProbeDepth;
}

namespace TB = Tablebases;

using std::string;
using Eval::evaluate;
using namespace Search;

namespace {

  // Different node types, used as a template parameter
  enum NodeType { NonPV, PV };

  constexpr uint64_t TtHitAverageWindow     = 4096;
  constexpr uint64_t TtHitAverageResolution = 1024;

  // Razor and futility margins
  constexpr int RazorMargin = 527;
  Value futility_margin(Depth d, bool improving) {
    return Value(227 * (d - improving));
  }

  // Reductions lookup table, initialized at startup
  int Reductions[MAX_MOVES]; // [depth or moveNumber]

  Depth reduction(bool i, Depth d, int mn) {
    int r = Reductions[d] * Reductions[mn];
    return (r + 570) / 1024 + (!i && r > 1018);
  }

  constexpr int futility_move_count(bool improving, Depth depth) {
    return (3 + depth * depth) / (2 - improving);
  }

  // History and stats update bonus, based on depth
  int stat_bonus(Depth d) {
    return d > 15 ? 27 : 17 * d * d + 133 * d - 134;
  }

  // Add a small random component to draw evaluations to avoid 3fold-blindness
  Value value_draw(Thread* thisThread) {
    return VALUE_DRAW + Value(2 * (thisThread->nodes & 1) - 1);
  }

  // Skill structure is used to implement strength limit
  struct Skill {
    explicit Skill(int l) : level(l) {}
    bool enabled() const { return level < 20; }
    bool time_to_pick(Depth depth) const { return depth == 1 + level; }
    Move pick_best(size_t multiPV);

    int level;
    Move best = MOVE_NONE;
  };

  // Breadcrumbs are used to mark nodes as being searched by a given thread
  struct Breadcrumb {
    std::atomic<Thread*> thread;
    std::atomic<Key> key;
  };
  std::array<Breadcrumb, 1024> breadcrumbs;

  // ThreadHolding structure keeps track of which thread left breadcrumbs at the given
  // node for potential reductions.
  struct ThreadHolding {
    explicit ThreadHolding(Thread* thisThread, Key posKey, int ply) {
       location = ply < 8 ? &breadcrumbs[posKey & (breadcrumbs.size() - 1)] : nullptr;
       otherThread = false;
       owning = false;
       if (location)
       {
          Thread* tmp = (*location).thread.load(std::memory_order_relaxed);
          if (tmp == nullptr)
          {
              (*location).thread.store(thisThread, std::memory_order_relaxed);
              (*location).key.store(posKey, std::memory_order_relaxed);
              owning = true;
          }
          else if (   tmp != thisThread
                   && (*location).key.load(std::memory_order_relaxed) == posKey)
              otherThread = true;
       }
    }

    ~ThreadHolding() {
       if (owning)
           (*location).thread.store(nullptr, std::memory_order_relaxed);
    }

    bool marked() { return otherThread; }

    private:
    Breadcrumb* location;
    bool otherThread, owning;
  };

  template <NodeType NT>
  Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

  template <NodeType NT>
  Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth = 0);

  Value value_to_tt(Value v, int ply);
  Value value_from_tt(Value v, int ply, int r50c);
  void update_pv(Move* pv, Move move, Move* childPv);
  void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
  void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus, int depth);
  void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
                        Move* quietsSearched, int quietCount, Move* capturesSearched, int captureCount, Depth depth);

  template<bool Root>
  uint64_t perft(Position& pos, Depth depth) {

    StateInfo st;
    uint64_t cnt, nodes = 0;
    const bool leaf = (depth == 2);

    for (const auto& m : MoveList<LEGAL>(pos))
    {
        if (Root && depth <= 1)
            cnt = 1, nodes++;
        else
        {
            pos.do_move(m, st);
            cnt = leaf ? MoveList<LEGAL>(pos).size() : perft<false>(pos, depth - 1);
            nodes += cnt;
            pos.undo_move(m);
        }
        if (Root)
            sync_cout << UCI::move(m, pos.is_chess960()) << ": " << cnt << sync_endl;
    }
    return nodes;
  }

} // namespace


void Search::init() {
  for (int i = 1; i < MAX_MOVES; ++i)
      Reductions[i] = int((24.8 + std::log(Threads.size())) * std::log(i));
}

void Search::clear() {
  Threads.main()->wait_for_search_finished();
  Time.availableNodes = 0;
  TT.clear();
  Threads.clear();
  Tablebases::init(Options["SyzygyPath"]);
}

void MainThread::search() {
  if (Limits.perft)
  {
      nodes = perft<true>(rootPos, Limits.perft);
      sync_cout << "\nNodes searched: " << nodes << "\n" << sync_endl;
      return;
  }

  Color us = rootPos.side_to_move();
  Time.init(Limits, us, rootPos.game_ply());
  TT.new_search();

  if (rootMoves.empty())
  {
      rootMoves.emplace_back(MOVE_NONE);
      sync_cout << "info depth 0 score "
                << UCI::value(rootPos.checkers() ? -VALUE_MATE : VALUE_DRAW)
                << sync_endl;
  }
  else
  {
      Threads.start_searching();
      Thread::search();
  }

  while (!Threads.stop && (ponder || Limits.infinite))
  {}

  Threads.stop = true;
  Threads.wait_for_search_finished();

  if (Limits.npmsec)
      Time.availableNodes += Limits.inc[us] - Threads.nodes_searched();

  Thread* bestThread = this;

  if (   int(Options["MultiPV"]) == 1
      && !Limits.depth
      && !(Skill(Options["Skill Level"]).enabled() || int(Options["UCI_LimitStrength"]))
      && rootMoves[0].pv[0] != MOVE_NONE)
      bestThread = Threads.get_best_thread();

  bestPreviousScore = bestThread->rootMoves[0].score;

  if (bestThread != this)
      sync_cout << UCI::pv(bestThread->rootPos, bestThread->completedDepth, -VALUE_INFINITE, VALUE_INFINITE) << sync_endl;

  sync_cout << "bestmove " << UCI::move(bestThread->rootMoves[0].pv[0], rootPos.is_chess960());

  if (bestThread->rootMoves[0].pv.size() > 1 || bestThread->rootMoves[0].extract_ponder_from_tt(rootPos))
      std::cout << " ponder " << UCI::move(bestThread->rootMoves[0].pv[1], rootPos.is_chess960());

  std::cout << sync_endl;
}

void Thread::search() {
  Stack stack[MAX_PLY+10], *ss = stack+7;
  Move  pv[MAX_PLY+1];
  Value bestValue, alpha, beta, delta;
  Move  lastBestMove = MOVE_NONE;
  Depth lastBestMoveDepth = 0;
  MainThread* mainThread = (this == Threads.main() ? Threads.main() : nullptr);
  double timeReduction = 1, totBestMoveChanges = 0;
  Color us = rootPos.side_to_move();
  int iterIdx = 0;

  std::memset(ss-7, 0, 10 * sizeof(Stack));
  for (int i = 7; i > 0; i--)
      (ss-i)->continuationHistory = &this->continuationHistory[0][0][NO_PIECE][0];

  ss->pv = pv;

  bestValue = delta = alpha = -VALUE_INFINITE;
  beta = VALUE_INFINITE;

  if (mainThread)
  {
      if (mainThread->bestPreviousScore == VALUE_INFINITE)
          for (int i = 0; i < 4; ++i)
              mainThread->iterValue[i] = VALUE_ZERO;
      else
          for (int i = 0; i < 4; ++i)
              mainThread->iterValue[i] = mainThread->bestPreviousScore;
  }

  std::copy(&lowPlyHistory[2][0], &lowPlyHistory.back().back() + 1, &lowPlyHistory[0][0]);
  std::fill(&lowPlyHistory[MAX_LPH - 2][0], &lowPlyHistory.back().back() + 1, 0);

  size_t multiPV = size_t(Options["MultiPV"]);
  PRNG rng(now());
  double floatLevel = Options["UCI_LimitStrength"] ?
                      Utility::clamp(std::pow((Options["UCI_Elo"] - 1346.6) / 143.4, 1 / 0.806), 0.0, 20.0) :
                        double(Options["Skill Level"]);
  int intLevel = int(floatLevel) +
                 ((floatLevel - int(floatLevel)) * 1024 > rng.rand<unsigned>() % 1024  ? 1 : 0);
  Skill skill(intLevel);

  if (skill.enabled())
      multiPV = std::max(multiPV, (size_t)4);

  multiPV = std::min(multiPV, rootMoves.size());
  ttHitAverage = TtHitAverageWindow * TtHitAverageResolution / 2;

  int ct = int(Options["Contempt"]) * PawnValueEg / 100;

  if (Limits.infinite || Options["UCI_AnalyseMode"])
      ct =  Options["Analysis Contempt"] == "Off"  ? 0
          : Options["Analysis Contempt"] == "Both" ? ct
          : Options["Analysis Contempt"] == "White" && us == BLACK ? -ct
          : Options["Analysis Contempt"] == "Black" && us == WHITE ? -ct
          : ct;

  contempt = (us == WHITE ?  make_score(ct, ct / 2)
                          : -make_score(ct, ct / 2));

  int searchAgainCounter = 0;

  while (   ++rootDepth < MAX_PLY
         && !Threads.stop
         && !(Limits.depth && mainThread && rootDepth > Limits.depth))
  {
      if (mainThread)
          totBestMoveChanges /= 2;

      for (RootMove& rm : rootMoves)
          rm.previousScore = rm.score;

      size_t pvFirst = 0;
      pvLast = 0;

      if (!Threads.increaseDepth)
         searchAgainCounter++;

      for (pvIdx = 0; pvIdx < multiPV && !Threads.stop; ++pvIdx)
      {
          if (pvIdx == pvLast)
          {
              pvFirst = pvLast;
              for (pvLast++; pvLast < rootMoves.size(); pvLast++)
                  if (rootMoves[pvLast].tbRank != rootMoves[pvFirst].tbRank)
                      break;
          }

          selDepth = 0;

          if (rootDepth >= 4)
          {
              Value prev = rootMoves[pvIdx].previousScore;
              delta = Value(19);
              alpha = std::max(prev - delta,-VALUE_INFINITE);
              beta  = std::min(prev + delta, VALUE_INFINITE);

              int dct = ct + (110 - ct / 2) * prev / (abs(prev) + 140);

              contempt = (us == WHITE ?  make_score(dct, dct / 2)
                                      : -make_score(dct, dct / 2));
          }

          int failedHighCnt = 0;
          while (true)
          {
              Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - searchAgainCounter);
              bestValue = ::search<PV>(rootPos, ss, alpha, beta, adjustedDepth, false);

              std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);

              if (Threads.stop)
                  break;

              if (   mainThread
                  && multiPV == 1
                  && (bestValue <= alpha || bestValue >= beta)
                  && Time.elapsed() > 3000)
                  sync_cout << UCI::pv(rootPos, rootDepth, alpha, beta) << sync_endl;

              if (bestValue <= alpha)
              {
                  beta = (alpha + beta) / 2;
                  alpha = std::max(bestValue - delta, -VALUE_INFINITE);

                  failedHighCnt = 0;
                  if (mainThread)
                      mainThread->stopOnPonderhit = false;
              }
              else if (bestValue >= beta)
              {
                  beta = std::min(bestValue + delta, VALUE_INFINITE);
                  ++failedHighCnt;
              }
              else
              {
                  ++rootMoves[pvIdx].bestMoveCount;
                  break;
              }

              delta += delta / 4 + 5;

              assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
          }

          std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);

          if (    mainThread
              && (Threads.stop || pvIdx + 1 == multiPV || Time.elapsed() > 3000))
              sync_cout << UCI::pv(rootPos, rootDepth, alpha, beta) << sync_endl;
      }

      if (!Threads.stop)
          completedDepth = rootDepth;

      if (rootMoves[0].pv[0] != lastBestMove) {
         lastBestMove = rootMoves[0].pv[0];
         lastBestMoveDepth = rootDepth;
      }

      if (   Limits.mate
          && bestValue >= VALUE_MATE_IN_MAX_PLY
          && VALUE_MATE - bestValue <= 2 * Limits.mate)
          Threads.stop = true;

      if (!mainThread)
          continue;

      if (skill.enabled() && skill.time_to_pick(rootDepth))
          skill.pick_best(multiPV);

      if (    Limits.use_time_management()
          && !Threads.stop
          && !mainThread->stopOnPonderhit)
      {
          double fallingEval = (296 + 6 * (mainThread->bestPreviousScore - bestValue)
                                    + 6 * (mainThread->iterValue[iterIdx] - bestValue)) / 725.0;
          fallingEval = Utility::clamp(fallingEval, 0.5, 1.5);

          timeReduction = lastBestMoveDepth + 10 < completedDepth ? 1.92 : 0.95;
          double reduction = (1.47 + mainThread->previousTimeReduction) / (2.22 * timeReduction);

          for (Thread* th : Threads)
          {
              totBestMoveChanges += th->bestMoveChanges;
              th->bestMoveChanges = 0;
          }
          double bestMoveInstability = 1 + totBestMoveChanges / Threads.size();

          double totalTime = rootMoves.size() == 1 ? 0 :
                             Time.optimum() * fallingEval * reduction * bestMoveInstability;

          if (Time.elapsed() > totalTime)
          {
              if (mainThread->ponder)
                  mainThread->stopOnPonderhit = true;
              else
                  Threads.stop = true;
          }
          else if (   Threads.increaseDepth
                   && !mainThread->ponder
                   && Time.elapsed() > totalTime * 0.56)
                   Threads.increaseDepth = false;
          else
                   Threads.increaseDepth = true;
      }

      mainThread->iterValue[iterIdx] = bestValue;
      iterIdx = (iterIdx + 1) & 3;
  }

  if (!mainThread)
      return;

  mainThread->previousTimeReduction = timeReduction;

  if (skill.enabled())
      std::swap(rootMoves[0], *std::find(rootMoves.begin(), rootMoves.end(),
                skill.best ? skill.best : skill.pick_best(multiPV)));
}

namespace {

  template <NodeType NT>
  Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {

    constexpr bool PvNode = NT == PV;
    const bool rootNode = PvNode && ss->ply == 0;

    if (   pos.rule50_count() >= 3
        && alpha < VALUE_DRAW
        && !rootNode
        && pos.has_game_cycle(ss->ply))
    {
        alpha = value_draw(pos.this_thread());
        if (alpha >= beta)
            return alpha;
    }

    if (depth <= 0)
        return qsearch<NT>(pos, ss, alpha, beta);

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));

    Move pv[MAX_PLY+1], capturesSearched[32], quietsSearched[64];
    StateInfo st;
    TTEntry* tte;
    Key posKey;
    Move ttMove, move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, ttValue, eval, maxValue, probcutBeta;
    bool ttHit, ttPv, formerPv, givesCheck, improving, didLMR, priorCapture;
    bool captureOrPromotion, doFullDepthSearch, moveCountPruning,
         ttCapture, singularQuietLMR;
    Piece movedPiece;
    int moveCount, captureCount, quietCount;

    // Step 1. Initialize node
    Thread* thisThread = pos.this_thread();
    ss->inCheck = pos.checkers();
    priorCapture = pos.captured_piece();
    Color us = pos.side_to_move();
    moveCount = captureCount = quietCount = ss->moveCount = 0;
    bestValue = -VALUE_INFINITE;
    maxValue = VALUE_INFINITE;

    if (thisThread == Threads.main())
        static_cast<MainThread*>(thisThread)->check_time();

    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;

    if (!rootNode)
    {
        if (   Threads.stop.load(std::memory_order_relaxed)
            || pos.is_draw(ss->ply)
            || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos)
                                                        : value_draw(pos.this_thread());

        alpha = std::max(mated_in(ss->ply), alpha);
        beta = std::min(mate_in(ss->ply+1), beta);
        if (alpha >= beta)
            return alpha;
    }

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    (ss+1)->ply = ss->ply + 1;
    (ss+1)->excludedMove = bestMove = MOVE_NONE;
    (ss+2)->killers[0] = (ss+2)->killers[1] = MOVE_NONE;
    Square prevSq = to_sq((ss-1)->currentMove);

    if (rootNode)
        (ss+4)->statScore = 0;
    else
        (ss+2)->statScore = 0;

    excludedMove = ss->excludedMove;
    posKey = excludedMove == MOVE_NONE ? pos.key() : pos.key() ^ make_key(excludedMove);
    tte = TT.probe(posKey, ttHit);
    ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
    ttMove =  rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0]
            : ttHit    ? tte->move() : MOVE_NONE;
    ttPv = PvNode || (ttHit && tte->is_pv());
    formerPv = ttPv && !PvNode;

    if (   ttPv
        && depth > 12
        && ss->ply - 1 < MAX_LPH
        && !priorCapture
        && is_ok((ss-1)->currentMove))
        thisThread->lowPlyHistory[ss->ply - 1][from_to((ss-1)->currentMove)] << stat_bonus(depth - 5);

    thisThread->ttHitAverage =   (TtHitAverageWindow - 1) * thisThread->ttHitAverage / TtHitAverageWindow
                                + TtHitAverageResolution * ttHit;

    if (  !PvNode
        && ttHit
        && tte->depth() >= depth
        && ttValue != VALUE_NONE
        && (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
                            : (tte->bound() & BOUND_UPPER)))
    {
        if (ttMove)
        {
            if (ttValue >= beta)
            {
                if (!pos.capture_or_promotion(ttMove))
                    update_quiet_stats(pos, ss, ttMove, stat_bonus(depth), depth);

                if ((ss-1)->moveCount <= 2 && !priorCapture)
                    update_continuation_histories(ss-1, pos.piece_on(prevSq), prevSq, -stat_bonus(depth + 1));
            }
            else if (!pos.capture_or_promotion(ttMove))
            {
                int penalty = -stat_bonus(depth);
                thisThread->mainHistory[us][from_to(ttMove)] << penalty;
                update_continuation_histories(ss, pos.moved_piece(ttMove), to_sq(ttMove), penalty);
            }
        }

        if (pos.rule50_count() < 90)
            return ttValue;
    }

    if (!rootNode && TB::Cardinality)
    {
        int piecesCount = pos.count<ALL_PIECES>();

        if (    piecesCount <= TB::Cardinality
            && (piecesCount <  TB::Cardinality || depth >= TB::ProbeDepth)
            &&  pos.rule50_count() == 0
            && !pos.can_castle(ANY_CASTLING))
        {
            TB::ProbeState err;
            TB::WDLScore wdl = Tablebases::probe_wdl(pos, &err);

            if (thisThread == Threads.main())
                static_cast<MainThread*>(thisThread)->callsCnt = 0;

            if (err != TB::ProbeState::FAIL)
            {
                thisThread->tbHits.fetch_add(1, std::memory_order_relaxed);

                int drawScore = TB::UseRule50 ? 1 : 0;

                value =  wdl < -drawScore ? VALUE_MATED_IN_MAX_PLY + ss->ply + 1
                       : wdl >  drawScore ? VALUE_MATE_IN_MAX_PLY - ss->ply - 1
                                          : VALUE_DRAW + 2 * wdl * drawScore;

                Bound b =  wdl < -drawScore ? BOUND_UPPER
                         : wdl >  drawScore ? BOUND_LOWER : BOUND_EXACT;

                if (    b == BOUND_EXACT
                    || (b == BOUND_LOWER ? value >= beta : value <= alpha))
                {
                    tte->save(posKey, value_to_tt(value, ss->ply), ttPv, b,
                              std::min(MAX_PLY - 1, depth + 6),
                              MOVE_NONE, VALUE_NONE);

                    return value;
                }

                if (PvNode)
                {
                    if (b == BOUND_LOWER)
                        bestValue = value, alpha = std::max(alpha, bestValue);
                    else
                        maxValue = value;
                }
            }
        }
    }

    CapturePieceToHistory& captureHistory = thisThread->captureHistory;

    if (ss->inCheck)
    {
        ss->staticEval = eval = VALUE_NONE;
        improving = false;
        goto moves_loop;
    }
    else if (ttHit)
    {
        ss->staticEval = eval = tte->eval();
        if (eval == VALUE_NONE)
            ss->staticEval = eval = evaluate(pos);

        if (eval == VALUE_DRAW)
            eval = value_draw(thisThread);

        if (    ttValue != VALUE_NONE
            && (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttValue;
    }
    else
    {
        if ((ss-1)->currentMove != MOVE_NULL)
        {
            int bonus = -(ss-1)->statScore / 512;
            ss->staticEval = eval = evaluate(pos) + bonus;
        }
        else
            ss->staticEval = eval = -(ss-1)->staticEval + 2 * Tempo;

        tte->save(posKey, VALUE_NONE, ttPv, BOUND_NONE, DEPTH_NONE, MOVE_NONE, eval);
    }

    if (   !rootNode
        &&  depth == 1
        &&  eval <= alpha - RazorMargin)
        return qsearch<NT>(pos, ss, alpha, beta);

    improving =  (ss-2)->staticEval == VALUE_NONE ? (ss->staticEval > (ss-4)->staticEval
              || (ss-4)->staticEval == VALUE_NONE) : ss->staticEval > (ss-2)->staticEval;

    if (   !PvNode
        &&  depth < 6
        &&  eval - futility_margin(depth, improving) >= beta
        &&  eval < VALUE_KNOWN_WIN)
        return eval;

    if (   !PvNode
        && (ss-1)->currentMove != MOVE_NULL
        && (ss-1)->statScore < 23824
        &&  eval >= beta
        &&  eval >= ss->staticEval
        &&  ss->staticEval >= beta - 33 * depth - 33 * improving + 112 * ttPv + 311
        && !excludedMove
        &&  pos.non_pawn_material(us)
        && (ss->ply >= thisThread->nmpMinPly || us != thisThread->nmpColor))
    {
        assert(eval - beta >= 0);

        Depth R = (737 + 77 * depth) / 246 + std::min(int(eval - beta) / 192, 3);

        ss->currentMove = MOVE_NULL;
        ss->continuationHistory = &thisThread->continuationHistory[0][0][NO_PIECE][0];

        pos.do_null_move(st);

        Value nullValue = -search<NonPV>(pos, ss+1, -beta, -beta+1, depth-R, !cutNode);

        pos.undo_null_move();

        if (nullValue >= beta)
        {
            if (nullValue >= VALUE_TB_WIN_IN_MAX_PLY)
                nullValue = beta;

            if (thisThread->nmpMinPly || (abs(beta) < VALUE_KNOWN_WIN && depth < 13))
                return nullValue;

            thisThread->nmpMinPly = ss->ply + 3 * (depth-R) / 4;
            thisThread->nmpColor = us;

            Value v = search<NonPV>(pos, ss, beta-1, beta, depth-R, false);

            thisThread->nmpMinPly = 0;

            if (v >= beta)
                return nullValue;
        }
    }

    probcutBeta = beta + 176 - 49 * improving;

    if (   !PvNode
        &&  depth > 4
        &&  abs(beta) < VALUE_TB_WIN_IN_MAX_PLY
        && !(   ttHit
             && tte->depth() >= depth - 3
             && ttValue != VALUE_NONE
             && ttValue < probcutBeta))
    {
        if (   ttHit
            && tte->depth() >= depth - 3
            && ttValue != VALUE_NONE
            && ttValue >= probcutBeta
            && ttMove
            && pos.capture_or_promotion(ttMove))
            return probcutBeta;

        assert(probcutBeta < VALUE_INFINITE);
        MovePicker mp(pos, ttMove, probcutBeta - ss->staticEval, &captureHistory);
        int probCutCount = 0;

        while (   (move = mp.next_move()) != MOVE_NONE
               && probCutCount < 2 + 2 * cutNode)
            if (move != excludedMove && pos.legal(move))
            {
                assert(pos.capture_or_promotion(move));
                assert(depth >= 5);

                captureOrPromotion = true;
                probCutCount++;

                ss->currentMove = move;
                ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
                                                                          [captureOrPromotion]
                                                                          [pos.moved_piece(move)]
                                                                          [to_sq(move)];

                pos.do_move(move, st);

                value = -qsearch<NonPV>(pos, ss+1, -probcutBeta, -probcutBeta+1);

                if (value >= probcutBeta)
                    value = -search<NonPV>(pos, ss+1, -probcutBeta, -probcutBeta+1, depth - 4, !cutNode);

                pos.undo_move(move);

                if (value >= probcutBeta)
                {
                    if ( !(ttHit
                       && tte->depth() >= depth - 3
                       && ttValue != VALUE_NONE))
                        tte->save(posKey, value_to_tt(value, ss->ply), ttPv,
                            BOUND_LOWER,
                            depth - 3, move, ss->staticEval);
                    return value;
                }
            }
    }

    if (depth >= 7 && !ttMove)
    {
        search<NT>(pos, ss, alpha, beta, depth - 7, cutNode);

        tte = TT.probe(posKey, ttHit);
        ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
        ttMove = ttHit ? tte->move() : MOVE_NONE;
    }

moves_loop:

    const PieceToHistory* contHist[] = { (ss-1)->continuationHistory, (ss-2)->continuationHistory,
                                          nullptr                   , (ss-4)->continuationHistory,
                                          nullptr                   , (ss-6)->continuationHistory };

    Move countermove = thisThread->counterMoves[pos.piece_on(prevSq)][prevSq];

    MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory,
                                      &thisThread->lowPlyHistory,
                                      &captureHistory,
                                      contHist,
                                      countermove,
                                      ss->killers,
                                      ss->ply);

    value = bestValue;
    singularQuietLMR = moveCountPruning = false;
    ttCapture = ttMove && pos.capture_or_promotion(ttMove);

    ThreadHolding th(thisThread, posKey, ss->ply);

    while ((move = mp.next_move(moveCountPruning)) != MOVE_NONE)
    {
      assert(is_ok(move));

      if (move == excludedMove)
          continue;

      if (rootNode && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
                                  thisThread->rootMoves.begin() + thisThread->pvLast, move))
          continue;

      ss->moveCount = ++moveCount;

      if (rootNode && thisThread == Threads.main() && Time.elapsed() > 3000)
          sync_cout << "info depth " << depth
                    << " currmove " << UCI::move(move, pos.is_chess960())
                    << " currmovenumber " << moveCount + thisThread->pvIdx << sync_endl;
      if (PvNode)
          (ss+1)->pv = nullptr;

      extension = 0;
      captureOrPromotion = pos.capture_or_promotion(move);
      movedPiece = pos.moved_piece(move);
      givesCheck = pos.gives_check(move);

      newDepth = depth - 1;

      if (  !rootNode
          && pos.non_pawn_material(us)
          && bestValue > VALUE_TB_LOSS_IN_MAX_PLY)
      {
          moveCountPruning = moveCount >= futility_move_count(improving, depth);

          int lmrDepth = std::max(newDepth - reduction(improving, depth, moveCount), 0);

          if (   !captureOrPromotion
              && !givesCheck)
          {
              if (   lmrDepth < 4 + ((ss-1)->statScore > 0 || (ss-1)->moveCount == 1)
                  && (*contHist[0])[movedPiece][to_sq(move)] < CounterMovePruneThreshold
                  && (*contHist[1])[movedPiece][to_sq(move)] < CounterMovePruneThreshold)
                  continue;

              if (   lmrDepth < 6
                  && !ss->inCheck
                  && ss->staticEval + 284 + 188 * lmrDepth <= alpha
                  &&  (*contHist[0])[movedPiece][to_sq(move)]
                    + (*contHist[1])[movedPiece][to_sq(move)]
                    + (*contHist[3])[movedPiece][to_sq(move)]
                    + (*contHist[5])[movedPiece][to_sq(move)] / 2 < 28388)
                  continue;

              if (!pos.see_ge(move, Value(-(29 - std::min(lmrDepth, 17)) * lmrDepth * lmrDepth)))
                  continue;
          }
          else
          {
              if (   !givesCheck
                  && lmrDepth < 1
                  && captureHistory[movedPiece][to_sq(move)][type_of(pos.piece_on(to_sq(move)))] < 0)
                  continue;

              if (   !givesCheck
                  && lmrDepth < 6
                  && !(PvNode && abs(bestValue) < 2)
                  && PieceValue[MG][type_of(movedPiece)] >= PieceValue[MG][type_of(pos.piece_on(to_sq(move)))]
                  && !ss->inCheck
                  && ss->staticEval + 267 + 391 * lmrDepth
                     + PieceValue[MG][type_of(pos.piece_on(to_sq(move)))] <= alpha)
                  continue;

              if (!pos.see_ge(move, Value(-202) * depth))
                  continue;
          }
      }

      // Step 14. Extensions (Modified for aggressive search)
      if (    depth >= 6
          &&  move == ttMove
          && !rootNode
          && !excludedMove
          &&  abs(ttValue) < VALUE_KNOWN_WIN
          && (tte->bound() & BOUND_LOWER)
          &&  tte->depth() >= depth - 3
          &&  pos.legal(move))
      {
          Value singularBeta = ttValue - ((formerPv + 4) * depth) / 2;
          Depth singularDepth = (depth - 1 + 3 * formerPv) / 2;
          ss->excludedMove = move;
          value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
          ss->excludedMove = MOVE_NONE;

          if (value < singularBeta)
          {
              extension = 1;
              singularQuietLMR = !ttCapture;
          }
          else if (singularBeta >= beta)
              return singularBeta;
          else if (ttValue >= beta)
          {
              ss->excludedMove = move;
              value = search<NonPV>(pos, ss, beta - 1, beta, (depth + 3) / 2, cutNode);
              ss->excludedMove = MOVE_NONE;

              if (value >= beta)
                  return beta;
          }
      }
      // **Modifikasi**: Selalu beri ekstensi untuk langkah yang memberikan cek
      else if (givesCheck)
          extension = 1;
      else if (   move == ss->killers[0]
               && pos.advanced_pawn_push(move)
               && pos.pawn_passed(us, to_sq(move)))
          extension = 1;
      else if (   PieceValue[EG][pos.captured_piece()] > PawnValueEg
               && pos.non_pawn_material() <= 2 * RookValueMg)
          extension = 1;
      else if (type_of(move) == CASTLING)
          extension = 1;

      newDepth += extension;

      prefetch(TT.first_entry(pos.key_after(move)));

      if (!rootNode && !pos.legal(move))
      {
          ss->moveCount = --moveCount;
          continue;
      }

      ss->currentMove = move;
      ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
                                                                [captureOrPromotion]
                                                                [movedPiece]
                                                                [to_sq(move)];

      pos.do_move(move, st, givesCheck);

      // Step 16. Reduced depth search (LMR) (Modified for aggressive search)
      if (    depth >= 3
          &&  moveCount > 1 + 2 * rootNode
          && (!rootNode || thisThread->best_move_count(move) == 0)
          && (  !captureOrPromotion
              || moveCountPruning
              || ss->staticEval + PieceValue[EG][pos.captured_piece()] <= alpha
              || cutNode
              || thisThread->ttHitAverage < 415 * TtHitAverageResolution * TtHitAverageWindow / 1024))
      {
          Depth r = reduction(improving, depth, moveCount);

          if (   cutNode
              && depth <= 10
              && moveCount <= 2
              && !ss->inCheck)
              r--;

          if (thisThread->ttHitAverage > 473 * TtHitAverageResolution * TtHitAverageWindow / 1024)
              r--;

          if (th.marked())
              r++;

          if (ttPv)
              r -= 2;

          if (moveCountPruning && !formerPv)
              r++;

          if ((ss-1)->moveCount > 13)
              r--;

          if (singularQuietLMR)
              r -= 1 + formerPv;

          if (!captureOrPromotion)
          {
              if (ttCapture)
                  r++;

              if (cutNode)
                  r += 2;

              else if (    type_of(move) == NORMAL
                       && !pos.see_ge(reverse_move(move)))
                  r -= 2 + ttPv - (type_of(movedPiece) == PAWN);

              ss->statScore =  thisThread->mainHistory[us][from_to(move)]
                             + (*contHist[0])[movedPiece][to_sq(move)]
                             + (*contHist[1])[movedPiece][to_sq(move)]
                             + (*contHist[3])[movedPiece][to_sq(move)]
                             - 4826;

              if (ss->statScore >= -100 && (ss-1)->statScore < -112)
                  r--;

              else if ((ss-1)->statScore >= -125 && ss->statScore < -138)
                  r++;

              r -= ss->statScore / 14615;
          }
          else
          {
            if (depth < 8 && moveCount > 2)
                r++;

            if (   !givesCheck
                && ss->staticEval + PieceValue[EG][pos.captured_piece()] + 211 * depth <= alpha)
                r++;
          }

          // **Modifikasi**: Kurangi reduksi untuk langkah agresif (capture atau check)
          if (captureOrPromotion || givesCheck)
              r -= 1;

          Depth d = Utility::clamp(newDepth - r, 1, newDepth);

          value = -search<NonPV>(pos, ss+1, -(alpha+1), -alpha, d, true);

          doFullDepthSearch = value > alpha && d != newDepth;

          didLMR = true;
      }
      else
      {
          doFullDepthSearch = !PvNode || moveCount > 1;

          didLMR = false;
      }

      if (doFullDepthSearch)
      {
          value = -search<NonPV>(pos, ss+1, -(alpha+1), -alpha, newDepth, !cutNode);

          if (didLMR && !captureOrPromotion)
          {
              int bonus = value > alpha ?  stat_bonus(newDepth)
                                        : -stat_bonus(newDepth);

              if (move == ss->killers[0])
                  bonus += bonus / 4;

              update_continuation_histories(ss, movedPiece, to_sq(move), bonus);
          }
      }

      if (PvNode && (moveCount == 1 || (value > alpha && (rootNode || value < beta))))
      {
          (ss+1)->pv = pv;
          (ss+1)->pv[0] = MOVE_NONE;

          value = -search<PV>(pos, ss+1, -beta, -alpha, newDepth, false);
      }

      pos.undo_move(move);

      assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

      if (Threads.stop.load(std::memory_order_relaxed))
          return VALUE_ZERO;

      if (rootNode)
      {
          RootMove& rm = *std::find(thisThread->rootMoves.begin(),
                                    thisThread->rootMoves.end(), move);

          if (moveCount == 1 || value > alpha)
          {
              rm.score = value;
              rm.selDepth = thisThread->selDepth;
              rm.pv.resize(1);

              assert((ss+1)->pv);

              for (Move* m = (ss+1)->pv; *m != MOVE_NONE; ++m)
                  rm.pv.push_back(*m);

              if (moveCount > 1)
                  ++thisThread->bestMoveChanges;
          }
          else
              rm.score = -VALUE_INFINITE;
      }

      if (value > bestValue)
      {
          bestValue = value;

          if (value > alpha)
          {
              bestMove = move;

              if (PvNode && !rootNode)
                  update_pv(ss->pv, move, (ss+1)->pv);

              if (PvNode && value < beta)
                  alpha = value;
              else
              {
                  assert(value >= beta);
                  ss->statScore = 0;
                  break;
              }
          }
      }

      if (move != bestMove)
      {
          if (captureOrPromotion && captureCount < 32)
              capturesSearched[captureCount++] = move;

          else if (!captureOrPromotion && quietCount < 64)
              quietsSearched[quietCount++] = move;
      }
    }

    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());

    if (!moveCount)
        bestValue = excludedMove ? alpha
                   :     ss->inCheck ? mated_in(ss->ply) : VALUE_DRAW;

    else if (bestMove)
        update_all_stats(pos, ss, bestMove, bestValue, beta, prevSq,
                         quietsSearched, quietCount, capturesSearched, captureCount, depth);

    else if (   (depth >= 3 || PvNode)
             && !priorCapture)
        update_continuation_histories(ss-1, pos.piece_on(prevSq), prevSq, stat_bonus(depth));

    if (PvNode)
        bestValue = std::min(bestValue, maxValue);

    if (!excludedMove && !(rootNode && thisThread->pvIdx))
        tte->save(posKey, value_to_tt(bestValue, ss->ply), ttPv,
                  bestValue >= beta ? BOUND_LOWER :
                  PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
                  depth, bestMove, ss->staticEval);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
  }

  template <NodeType NT>
  Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth) {

    constexpr bool PvNode = NT == PV;

    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(depth <= 0);

    Move pv[MAX_PLY+1];
    StateInfo st;
    TTEntry* tte;
    Key posKey;
    Move ttMove, move, bestMove;
    Depth ttDepth;
    Value bestValue, value, ttValue, futilityValue, futilityBase, oldAlpha;
    bool ttHit, pvHit, givesCheck, captureOrPromotion;
    int moveCount;

    if (PvNode)
    {
        oldAlpha = alpha;
        (ss+1)->pv = pv;
        ss->pv[0] = MOVE_NONE;
    }

    Thread* thisThread = pos.this_thread();
    (ss+1)->ply = ss->ply + 1;
    bestMove = MOVE_NONE;
    ss->inCheck = pos.checkers();
    moveCount = 0;

    if (   pos.is_draw(ss->ply)
        || ss->ply >= MAX_PLY)
        return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : VALUE_DRAW;

    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    ttDepth = ss->inCheck || depth >= DEPTH_QS_CHECKS ? DEPTH_QS_CHECKS
                                                  : DEPTH_QS_NO_CHECKS;

    posKey = pos.key();
    tte = TT.probe(posKey, ttHit);
    ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
    ttMove = ttHit ? tte->move() : MOVE_NONE;
    pvHit = ttHit && tte->is_pv();

    if (  !PvNode
        && ttHit
        && tte->depth() >= ttDepth
        && ttValue != VALUE_NONE
        && (ttValue >= beta ? (tte->bound() & BOUND_LOWER)
                            : (tte->bound() & BOUND_UPPER)))
        return ttValue;

    if (ss->inCheck)
    {
        ss->staticEval = VALUE_NONE;
        bestValue = futilityBase = -VALUE_INFINITE;
    }
    else
    {
        if (ttHit)
        {
            if ((ss->staticEval = bestValue = tte->eval()) == VALUE_NONE)
                ss->staticEval = bestValue = evaluate(pos);

            if (    ttValue != VALUE_NONE
                && (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttValue;
        }
        else
            ss->staticEval = bestValue =
            (ss-1)->currentMove != MOVE_NULL ? evaluate(pos)
                                             : -(ss-1)->staticEval + 2 * Tempo;

        if (bestValue >= beta)
        {
            if (!ttHit)
                tte->save(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                          DEPTH_NONE, MOVE_NONE, ss->staticEval);

            return bestValue;
        }

        if (PvNode && bestValue > alpha)
            alpha = bestValue;

        futilityBase = bestValue + 141;
    }

    const PieceToHistory* contHist[] = { (ss-1)->continuationHistory, (ss-2)->continuationHistory,
                                          nullptr                   , (ss-4)->continuationHistory,
                                          nullptr                   , (ss-6)->continuationHistory };

    MovePicker mp(pos, ttMove, depth, &thisThread->mainHistory,
                                      &thisThread->captureHistory,
                                      contHist,
                                      to_sq((ss-1)->currentMove));

    while ((move = mp.next_move()) != MOVE_NONE)
    {
      assert(is_ok(move));

      givesCheck = pos.gives_check(move);
      captureOrPromotion = pos.capture_or_promotion(move);

      moveCount++;

      if (   !ss->inCheck
          && !givesCheck
          &&  futilityBase > -VALUE_KNOWN_WIN
          && !pos.advanced_pawn_push(move))
      {
          assert(type_of(move) != ENPASSANT);

          futilityValue = futilityBase + PieceValue[EG][pos.piece_on(to_sq(move))];

          if (futilityValue <= alpha)
          {
              bestValue = std::max(bestValue, futilityValue);
              continue;
          }

          if (futilityBase <= alpha && !pos.see_ge(move, VALUE_ZERO + 1))
          {
              bestValue = std::max(bestValue, futilityBase);
              continue;
          }
      }

      if (  !ss->inCheck && !pos.see_ge(move))
          continue;

      prefetch(TT.first_entry(pos.key_after(move)));

      if (!pos.legal(move))
      {
          moveCount--;
          continue;
      }

      ss->currentMove = move;
      ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck]
                                                                [captureOrPromotion]
                                                                [pos.moved_piece(move)]
                                                                [to_sq(move)];

      pos.do_move(move, st, givesCheck);
      value = -qsearch<NT>(pos, ss+1, -beta, -alpha, depth - 1);
      pos.undo_move(move);

      assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

      if (value > bestValue)
      {
          bestValue = value;

          if (value > alpha)
          {
              bestMove = move;

              if (PvNode)
                  update_pv(ss->pv, move, (ss+1)->pv);

              if (PvNode && value < beta)
                  alpha = value;
              else
                  break;
          }
       }
    }

    if (ss->inCheck && bestValue == -VALUE_INFINITE)
        return mated_in(ss->ply);

    tte->save(posKey, value_to_tt(bestValue, ss->ply), pvHit,
              bestValue >= beta ? BOUND_LOWER :
              PvNode && bestValue > oldAlpha  ? BOUND_EXACT : BOUND_UPPER,
              ttDepth, bestMove, ss->staticEval);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);

    return bestValue;
  }

  Value value_to_tt(Value v, int ply) {
    assert(v != VALUE_NONE);
    return  v >= VALUE_TB_WIN_IN_MAX_PLY  ? v + ply
          : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply : v;
  }

  Value value_from_tt(Value v, int ply, int r50c) {
    if (v == VALUE_NONE)
        return VALUE_NONE;

    if (v >= VALUE_TB_WIN_IN_MAX_PLY)
    {
        if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 99 - r50c)
            return VALUE_MATE_IN_MAX_PLY - 1;
        return v - ply;
    }

    if (v <= VALUE_TB_LOSS_IN_MAX_PLY)
    {
        if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 99 - r50c)
            return VALUE_MATED_IN_MAX_PLY + 1;
        return v + ply;
    }

    return v;
  }

  void update_pv(Move* pv, Move move, Move* childPv) {
    for (*pv++ = move; childPv && *childPv != MOVE_NONE; )
        *pv++ = *childPv++;
    *pv = MOVE_NONE;
  }

  void update_all_stats(const Position& pos, Stack* ss, Move bestMove, Value bestValue, Value beta, Square prevSq,
                        Move* quietsSearched, int quietCount, Move* capturesSearched, int captureCount, Depth depth) {

    int bonus1, bonus2;
    Color us = pos.side_to_move();
    Thread* thisThread = pos.this_thread();
    CapturePieceToHistory& captureHistory = thisThread->captureHistory;
    Piece moved_piece = pos.moved_piece(bestMove);
    PieceType captured = type_of(pos.piece_on(to_sq(bestMove)));

    bonus1 = stat_bonus(depth + 1);
    bonus2 = bestValue > beta + PawnValueMg ? bonus1
                                            : stat_bonus(depth);

    if (!pos.capture_or_promotion(bestMove))
    {
        update_quiet_stats(pos, ss, bestMove, bonus2, depth);

        for (int i = 0; i < quietCount; ++i)
        {
            thisThread->mainHistory[us][from_to(quietsSearched[i])] << -bonus2;
            update_continuation_histories(ss, pos.moved_piece(quietsSearched[i]), to_sq(quietsSearched[i]), -bonus2);
        }
    }
    else
        captureHistory[moved_piece][to_sq(bestMove)][captured] << bonus1;

    if (   ((ss-1)->moveCount == 1 || ((ss-1)->currentMove == (ss-1)->killers[0]))
        && !pos.captured_piece())
            update_continuation_histories(ss-1, pos.piece_on(prevSq), prevSq, -bonus1);

    for (int i = 0; i < captureCount; ++i)
    {
        moved_piece = pos.moved_piece(capturesSearched[i]);
        captured = type_of(pos.piece_on(to_sq(capturesSearched[i])));
        captureHistory[moved_piece][to_sq(capturesSearched[i])][captured] << -bonus1;
    }
  }

  void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    for (int i : {1, 2, 4, 6})
    {
        if (ss->inCheck && i > 2)
            break;
        if (is_ok((ss-i)->currentMove))
            (*(ss-i)->continuationHistory)[pc][to] << bonus;
    }
  }

  void update_quiet_stats(const Position& pos, Stack* ss, Move move, int bonus, int depth) {
    if (ss->killers[0] != move)
    {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = move;
    }

    Color us = pos.side_to_move();
    Thread* thisThread = pos.this_thread();
    thisThread->mainHistory[us][from_to(move)] << bonus;
    update_continuation_histories(ss, pos.moved_piece(move), to_sq(move), bonus);

    if (type_of(pos.moved_piece(move)) != PAWN)
        thisThread->mainHistory[us][from_to(reverse_move(move))] << -bonus;

    if (is_ok((ss-1)->currentMove))
    {
        Square prevSq = to_sq((ss-1)->currentMove);
        thisThread->counterMoves[pos.piece_on(prevSq)][prevSq] = move;
    }

    if (depth > 11 && ss->ply < MAX_LPH)
        thisThread->lowPlyHistory[ss->ply][from_to(move)] << stat_bonus(depth - 6);
  }

  Move Skill::pick_best(size_t multiPV) {
    const RootMoves& rootMoves = Threads.main()->rootMoves;
    static PRNG rng(now());

    Value topScore = rootMoves[0].score;
    int delta = std::min(topScore - rootMoves[multiPV - 1].score, PawnValueMg);
    int weakness = 120 - 2 * level;
    int maxScore = -VALUE_INFINITE;

    for (size_t i = 0; i < multiPV; ++i)
    {
        int push = (  weakness * int(topScore - rootMoves[i].score)
                    + delta * (rng.rand<unsigned>() % weakness)) / 128;

        if (rootMoves[i].score + push >= maxScore)
        {
            maxScore = rootMoves[i].score + push;
            best = rootMoves[i].pv[0];
        }
    }

    return best;
  }

} // namespace

void MainThread::check_time() {
  if (--callsCnt > 0)
      return;

  callsCnt = Limits.nodes ? std::min(1024, int(Limits.nodes / 1024)) : 1024;

  static TimePoint lastInfoTime = now();

  TimePoint elapsed = Time.elapsed();
  TimePoint tick = Limits.startTime + elapsed;

  if (tick - lastInfoTime >= 1000)
  {
      lastInfoTime = tick;
      dbg_print();
  }

  if (ponder)
      return;

  if (   (Limits.use_time_management() && (elapsed > Time.maximum() - 10 || stopOnPonderhit))
      || (Limits.movetime && elapsed >= Limits.movetime)
      || (Limits.nodes && Threads.nodes_searched() >= (uint64_t)Limits.nodes))
      Threads.stop = true;
}

string UCI::pv(const Position& pos, Depth depth, Value alpha, Value beta) {
  std::stringstream ss;
  TimePoint elapsed = Time.elapsed() + 1;
  const RootMoves& rootMoves = pos.this_thread()->rootMoves;
  size_t pvIdx = pos.this_thread()->pvIdx;
  size_t multiPV = std::min((size_t)Options["MultiPV"], rootMoves.size());
  uint64_t nodesSearched = Threads.nodes_searched();
  uint64_t tbHits = Threads.tb_hits() + (TB::RootInTB ? rootMoves.size() : 0);

  for (size_t i = 0; i < multiPV; ++i)
  {
      bool updated = rootMoves[i].score != -VALUE_INFINITE;

      if (depth == 1 && !updated)
          continue;

      Depth d = updated ? depth : depth - 1;
      Value v = updated ? rootMoves[i].score : rootMoves[i].previousScore;

      bool tb = TB::RootInTB && abs(v) < VALUE_MATE_IN_MAX_PLY;
      v = tb ? rootMoves[i].tbScore : v;

      if (ss.rdbuf()->in_avail())
          ss << "\n";

      ss << "info"
         << " depth "    << d
         << " seldepth " << rootMoves[i].selDepth
         << " multipv "  << i + 1
         << " score "    << UCI::value(v);

      if (Options["UCI_ShowWDL"])
          ss << UCI::wdl(v, pos.game_ply());

      if (!tb && i == pvIdx)
          ss << (v >= beta ? " lowerbound" : v <= alpha ? " upperbound" : "");

      ss << " nodes "    << nodesSearched
         << " nps "      << nodesSearched * 1000 / elapsed;

      if (elapsed > 1000)
          ss << " hashfull " << TT.hashfull();

      ss << " tbhits "   << tbHits
         << " time "     << elapsed
         << " pv";

      for (Move m : rootMoves[i].pv)
          ss << " " << UCI::move(m, pos.is_chess960());
  }

  return ss.str();
}

bool RootMove::extract_ponder_from_tt(Position& pos) {
    StateInfo st;
    bool ttHit;

    assert(pv.size() == 1);

    if (pv[0] == MOVE_NONE)
        return false;

    pos.do_move(pv[0], st);
    TTEntry* tte = TT.probe(pos.key(), ttHit);

    if (ttHit)
    {
        Move m = tte->move();
        if (MoveList<LEGAL>(pos).contains(m))
            pv.push_back(m);
    }

    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

void Tablebases::rank_root_moves(Position& pos, Search::RootMoves& rootMoves) {
    RootInTB = false;
    UseRule50 = bool(Options["Syzygy50MoveRule"]);
    ProbeDepth = int(Options["SyzygyProbeDepth"]);
    Cardinality = int(Options["SyzygyProbeLimit"]);
    bool dtz_available = true;

    if (Cardinality > MaxCardinality)
    {
        Cardinality = MaxCardinality;
        ProbeDepth = 0;
    }

    if (Cardinality >= popcount(pos.pieces()) && !pos.can_castle(ANY_CASTLING))
    {
        RootInTB = root_probe(pos, rootMoves);

        if (!RootInTB)
        {
            dtz_available = false;
            RootInTB = root_probe_wdl(pos, rootMoves);
        }
    }

    if (RootInTB)
    {
        std::sort(rootMoves.begin(), rootMoves.end(),
                  [](const RootMove &a, const RootMove &b) { return a.tbRank > b.tbRank; } );

        if (dtz_available || rootMoves[0].tbScore <= VALUE_DRAW)
            Cardinality = 0;
    }
    else
    {
        for (auto& m : rootMoves)
            m.tbRank = 0;
    }
}
