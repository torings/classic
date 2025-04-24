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

  --- MODIFIED FOR AGGRESSIVE STYLE ---
  Modifications aim to prioritize captures (even slightly losing ones),
  historically good quiet moves, and checks in quiescence search more heavily.
  These changes likely REDUCE overall playing strength but alter playing style.
  Uses INTEGER ARITHMETIC for modifications.
  --- MODIFIED FOR AGGRESSIVE STYLE ---
*/

#include <cassert>
#include <algorithm>
#include <vector>

#include "movepick.h"
#include "movegen.h"
#include "types.h"

// Definisi nilai negatif besar untuk pengurutan skak di QSearch
#define LARGE_NEGATIVE_VALUE -30000

namespace {

  enum Stages {
    MAIN_TT, CAPTURE_INIT, GOOD_CAPTURE, REFUTATION, QUIET_INIT, QUIET, BAD_CAPTURE,
    EVASION_TT, EVASION_INIT, EVASION,
    PROBCUT_TT, PROBCUT_INIT, PROBCUT,
    QSEARCH_TT, QCHECK_INIT, QCAPTURE_INIT, QCAPTURE, QCHECK
  };

  // Fungsi untuk mengurutkan langkah secara parsial hingga batas tertentu
  void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {
    for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
        if (p->value >= limit)
        {
            ExtMove tmp = *p, *q;
            *p = *++sortedEnd;
            for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                *q = *(q - 1);
            *q = tmp;
        }
  }

} // namespace

// Konstruktor untuk pencarian utama
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh, const LowPlyHistory* lp,
                       const CapturePieceToHistory* cph, const PieceToHistory** ch, Move cm, const Move* killers, int pl)
           : pos(p), mainHistory(mh), lowPlyHistory(lp), captureHistory(cph), continuationHistory(ch),
             ttMove(ttm), refutations{{killers[0], 0}, {killers[1], 0}, {cm, 0}}, depth(d), ply(pl) {
  assert(d > 0);
  stage = (pos.checkers() ? EVASION_TT : MAIN_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// Konstruktor untuk pencarian quiescence
MovePicker::MovePicker(const Position& p, Move ttm, Depth d, const ButterflyHistory* mh,
                       const CapturePieceToHistory* cph, const PieceToHistory** ch, Square rs)
           : pos(p), mainHistory(mh), captureHistory(cph), continuationHistory(ch), ttMove(ttm), recaptureSquare(rs), depth(d) {
  assert(d <= 0);
  stage = (pos.checkers() ? EVASION_TT : QSEARCH_TT) + 
          !(ttm && (depth > DEPTH_QS_RECAPTURES || to_sq(ttm) == recaptureSquare) && pos.pseudo_legal(ttm));
}

// Konstruktor untuk ProbCut
MovePicker::MovePicker(const Position& p, Move ttm, Value th, const CapturePieceToHistory* cph)
           : pos(p), captureHistory(cph), ttMove(ttm), threshold(th) {
  assert(!pos.checkers());
  stage = PROBCUT_TT + !(ttm && pos.capture(ttm) && pos.pseudo_legal(ttm) && pos.see_ge(ttm, threshold));
}

// Fungsi untuk memberikan skor pada langkah berdasarkan tipe
template<GenType Type>
void MovePicker::score() {
  static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");
  for (auto& m : *this) {
      if (Type == CAPTURES) {
          // Meningkatkan bobot MVV dan riwayat penangkapan untuk agresivitas
          int mvvScore = int(PieceValue[MG][pos.piece_on(to_sq(m))]) * 8; // Dari 6 menjadi 8
          int historyScore = ((*captureHistory)[pos.moved_piece(m)][to_sq(m)][type_of(pos.piece_on(to_sq(m)))] * 3) / 2; // 1.5x
          m.value = mvvScore + historyScore;
      } 
      else if (Type == QUIETS) {
          // Meningkatkan bobot heuristik riwayat untuk langkah tenang
          int mainHistScore = ((*mainHistory)[pos.side_to_move()][from_to(m)] * 3) / 2; // 1.5x
          int contHist0 = 3 * (*continuationHistory[0])[pos.moved_piece(m)][to_sq(m)]; // Dari 2 menjadi 3
          int contHist1 = 3 * (*continuationHistory[1])[pos.moved_piece(m)][to_sq(m)]; // Dari 2 menjadi 3
          int contHist3 = 3 * (*continuationHistory[3])[pos.moved_piece(m)][to_sq(m)]; // Dari 2 menjadi 3
          int contHist5 = 2 * (*continuationHistory[5])[pos.moved_piece(m)][to_sq(m)]; // Dari 1 menjadi 2
          int lowPlyHistScore = (ply < MAX_LPH ? std::min(6, depth / 2) * (*lowPlyHistory)[ply][from_to(m)] : 0); // Skala lebih agresif
          m.value = mainHistScore + contHist0 + contHist1 + contHist3 + contHist5 + lowPlyHistScore;
      } 
      else { // EVASIONS
          if (pos.capture(m)) {
              // Meningkatkan nilai penangkapan saat penghindaran (~10% boost)
              Value baseValue = PieceValue[MG][pos.piece_on(to_sq(m))];
              Value attackerValue = Value(type_of(pos.moved_piece(m)));
              int baseValInt = int(baseValue);
              int boost = baseValInt / 10; // Boost 10%
              m.value = (baseValInt + boost) - int(attackerValue);
          } else {
              int historyScore = (*mainHistory)[pos.side_to_move()][from_to(m)]
                               + (*continuationHistory[0])[pos.moved_piece(m)][to_sq(m)];
              m.value = historyScore - (1 << 28); // Penalti besar tetap ada
          }
      }
  }
}

// Fungsi untuk memilih langkah berdasarkan predikat
template<MovePicker::PickType T, typename Pred>
Move MovePicker::select(Pred filter) {
  while (cur < endMoves) {
      if (T == Best)
          std::swap(*cur, *std::max_element(cur, endMoves));
      Move currentMove = (*cur).move;
      if (currentMove != ttMove && filter())
          return (*cur++).move;
      cur++;
  }
  return MOVE_NONE;
}

// Fungsi utama untuk mengambil langkah berikutnya
Move MovePicker::next_move(bool skipQuiets) {
top:
  switch (stage) {
  case MAIN_TT:
  case EVASION_TT:
  case QSEARCH_TT:
  case PROBCUT_TT:
      {
          Move TtMove = ttMove;
          ttMove = MOVE_NONE;
          ++stage;
          if (TtMove != MOVE_NONE) return TtMove;
          goto top;
      }
  case CAPTURE_INIT:
  case PROBCUT_INIT:
  case QCAPTURE_INIT:
      cur = endBadCaptures = moves;
      endMoves = generate<CAPTURES>(pos, cur);
      score<CAPTURES>();
      ++stage;
      goto top;
  case GOOD_CAPTURE:
      {
          // Menurunkan ambang SEE untuk memasukkan penangkapan yang sedikit merugikan
          Move bestGoodCapture = select<Best>([&](){
              return pos.see_ge(*cur, Value(-150 * cur->value / 1024)) ? true : (*endBadCaptures++ = *cur, false);
          });
          if (bestGoodCapture != MOVE_NONE)
              return bestGoodCapture;
      }
      refutationIdx = 0;
      numRefutations = 0;
      if (refutations[0].move != MOVE_NONE) validRefutations[numRefutations++] = refutations[0].move;
      if (refutations[1].move != MOVE_NONE && refutations[1].move != refutations[0].move) validRefutations[numRefutations++] = refutations[1].move;
      if (refutations[2].move != MOVE_NONE && refutations[2].move != refutations[0].move && refutations[2].move != refutations[1].move) validRefutations[numRefutations++] = refutations[2].move;
      ++stage;
      /* fallthrough */
  case REFUTATION:
      while (refutationIdx < numRefutations) {
          Move refMove = validRefutations[refutationIdx++];
          if (!pos.capture(refMove) && pos.pseudo_legal(refMove))
              return refMove;
      }
      ++stage;
      /* fallthrough */
  case QUIET_INIT:
      if (!skipQuiets) {
          cur = endBadCaptures;
          endMoves = generate<QUIETS>(pos, cur);
          score<QUIETS>();
          partial_insertion_sort(cur, endMoves, -3000 * depth);
      } else {
          cur = endBadCaptures;
          endMoves = endBadCaptures;
      }
      ++stage;
      /* fallthrough */
  case QUIET:
      {
          Move bestQuiet = select<Next>([&](){
              return cur->move != refutations[0].move && cur->move != refutations[1].move && cur->move != refutations[2].move;
          });
          if (bestQuiet != MOVE_NONE)
              return bestQuiet;
      }
      cur = moves;
      endMoves = endBadCaptures;
      ++stage;
      /* fallthrough */
  case BAD_CAPTURE:
      {
          Move badCap = select<Next>([](){ return true; });
          if (badCap != MOVE_NONE) return badCap;
      }
      return MOVE_NONE;
  case EVASION_INIT:
      cur = moves;
      endMoves = generate<EVASIONS>(pos, cur);
      score<EVASIONS>();
      ++stage;
      /* fallthrough */
  case EVASION:
      {
          Move evasionMove = select<Best>([](){ return true; });
          if (evasionMove != MOVE_NONE) return evasionMove;
      }
      return MOVE_NONE;
  case PROBCUT:
      {
          Move probCutMove = select<Best>([&](){ return pos.see_ge(*cur, threshold); });
          if (probCutMove != MOVE_NONE) return probCutMove;
      }
      return MOVE_NONE;
  case QCAPTURE:
      {
          Move qCap = select<Best>([&](){ return depth > DEPTH_QS_RECAPTURES || to_sq(cur->move) == recaptureSquare; });
          if (qCap != MOVE_NONE) return qCap;
      }
      if (depth != DEPTH_QS_CHECKS)
          return MOVE_NONE;
      ++stage;
      /* fallthrough */
  case QCHECK_INIT:
      cur = moves;
      endMoves = generate<QUIET_CHECKS>(pos, cur);
      // Menilai skak tenang berdasarkan riwayat untuk prioritas agresif
      if (cur < endMoves) {
          for (ExtMove* m_ptr = cur; m_ptr < endMoves; ++m_ptr) {
              ExtMove& m = *m_ptr;
              m.value = (*mainHistory)[pos.side_to_move()][from_to(m.move)]
                      + (*continuationHistory[0])[pos.moved_piece(m.move)][to_sq(m.move)];
          }
          partial_insertion_sort(cur, endMoves, LARGE_NEGATIVE_VALUE);
      }
      ++stage;
      /* fallthrough */
  case QCHECK:
      {
          // Memilih skak dengan skor terbaik untuk agresivitas
          Move qCheck = select<Best>([](){ return true; });
          if (qCheck != MOVE_NONE) return qCheck;
      }
      return MOVE_NONE;
  }
  assert(false);
  return MOVE_NONE;
}
