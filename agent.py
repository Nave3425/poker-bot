"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Goop"
AGENT_NAME = "PokerStink"


# ===== YOUR CODE HERE =====

# Fast strategy helpers.  This code intentionally uses only the standard library
# plus evaluator.py, which is already part of the tournament bundle.
import itertools as _itertools
import random as _random
import time as _time
from evaluator import get_table as _get_eval_table


# Index combinations are immutable and shared by every decision.
_HOLE_PAIRS = tuple(_itertools.combinations(range(5), 2))
_BOARD_TRIPLES = tuple(_itertools.combinations(range(5), 3))
_SHORT_DECK_STRAIGHTS = (
    0b000011111,  # 6-7-8-9-T
    0b000111110,  # 7-8-9-T-J
    0b001111100,  # 8-9-T-J-Q
    0b011111000,  # 9-T-J-Q-K
    0b111110000,  # T-J-Q-K-A
    (1 << 8) | 0b000001111,  # A-6-7-8-9 wheel
)
_FAST_TABLE = None


def _table():
    global _FAST_TABLE
    if _FAST_TABLE is None:
        _FAST_TABLE = _get_eval_table()
    return _FAST_TABLE


def _fallback_action(obs):
    """Always return a real legal action whenever the observation permits it."""
    legal = getattr(obs, "legal_actions", ()) or ()
    if Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    if legal:
        return legal[0]
    # The judge never asks for an action in a terminal state.  This final
    # branch is defensive only; the normal path is validated against legal.
    return Action.CHECK_CALL


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _preflop_quality(cards):
    """Fast 0..1 structural score for a five-card Short Deck PLO hand.

    It rewards multiple playable hole-card pairs, double-suitedness, high
    pairs, and connected rank coverage.  It is a ranking heuristic, not an
    asserted exact preflop-equity table.
    """
    if len(cards) != 5:
        return 0.50
    rank_counts = [0] * 9
    suit_counts = [0] * 4
    suit_rank_masks = [0] * 4
    rank_sum = 0
    high_count = 0
    for c in cards:
        if not 0 <= int(c) < 36:
            return 0.50
        r = int(c) >> 2
        s = int(c) & 3
        rank_counts[r] += 1
        suit_counts[s] += 1
        suit_rank_masks[s] |= 1 << r
        rank_sum += r
        if r >= 6:
            high_count += 1

    # High-card density is valuable, but less than in Hold'em because exactly
    # two hole cards must be used and the opponent also has five cards.
    q = 0.285 + 0.245 * (rank_sum / 40.0) + 0.075 * (high_count / 5.0)

    pair_ranks = 0
    pair_count = 0
    trips = 0
    quads = 0
    highest_pair = -1
    for r, n in enumerate(rank_counts):
        if n == 2:
            pair_count += 1
            highest_pair = max(highest_pair, r)
            pair_ranks += 1
        elif n == 3:
            trips += 1
            highest_pair = max(highest_pair, r)
        elif n == 4:
            quads += 1
            highest_pair = max(highest_pair, r)
    if pair_count:
        q += 0.055 + 0.012 * highest_pair
    if pair_count >= 2:
        q += 0.095
    if trips:
        q += 0.075 + 0.010 * highest_pair
    if quads:
        q += 0.15

    # Two distinct suited hole-card pairs are especially useful in PLO5.
    suited_groups = 0
    best_suit = -1
    ace_suited = False
    for s in range(4):
        n = suit_counts[s]
        if n >= 2:
            suited_groups += 1
            best_suit = max(best_suit, n)
            if suit_rank_masks[s] & (1 << 8):
                ace_suited = True
    if suited_groups >= 2:
        q += 0.135
    elif suited_groups == 1:
        q += 0.060
    if best_suit >= 3:
        q += 0.025
    if ace_suited:
        q += 0.045

    # Rank connectivity, including the Short Deck A-6-7-8-9 wheel.
    rank_mask = 0
    for r, n in enumerate(rank_counts):
        if n:
            rank_mask |= 1 << r
    best_window = 0
    for window in _SHORT_DECK_STRAIGHTS:
        best_window = max(best_window, (rank_mask & window).bit_count())
    if best_window >= 4:
        q += 0.090
    elif best_window == 3:
        q += 0.045

    # Small pairwise bonuses for adjacent and one-gap ranks, treating A-6 as
    # adjacent. This rewards connected options without overvaluing duplicate ranks.
    distinct = [r for r, n in enumerate(rank_counts) if n]
    adjacency = 0
    one_gap = 0
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            d = distinct[j] - distinct[i]
            if d == 1 or d == 8:  # A/6 wrap
                adjacency += 1
            elif d == 2:
                one_gap += 1
    q += min(adjacency, 4) * 0.008 + min(one_gap, 4) * 0.003
    return _clamp(q, 0.20, 0.94)


def _preflop_equity(cards):
    """Conservative heuristic proxy for equity versus an unobserved range."""
    q = _preflop_quality(cards)
    return _clamp(0.47 + (q - 0.54) * 0.62, 0.30, 0.72)


def _fast_board_masks(board):
    """Return the ten 3-card board masks for a completed five-card board."""
    bits = (1 << board[0], 1 << board[1], 1 << board[2],
            1 << board[3], 1 << board[4])
    return tuple(bits[i] | bits[j] | bits[k] for i, j, k in _BOARD_TRIPLES)


def _fast_omaha_score(hole, board_masks, table):
    """100 lookup-based Omaha evaluation without the allocations in omaha_score."""
    best = 0
    h0, h1, h2, h3, h4 = hole
    p0, p1, p2, p3, p4 = 1 << h0, 1 << h1, 1 << h2, 1 << h3, 1 << h4
    pair_masks = (p0 | p1, p0 | p2, p0 | p3, p0 | p4,
                  p1 | p2, p1 | p3, p1 | p4,
                  p2 | p3, p2 | p4, p3 | p4)
    for pm in pair_masks:
        for bm in board_masks:
            score = table[pm | bm]
            if score > best:
                best = score
    return best


def _opponent_range_weight(cards, profile):
    """Softly condition random deals on the opponent's observed preflop style."""
    kind = profile["kind"]
    if kind == "UNKNOWN" or profile["hands"] < 12:
        return 1.0
    q = _preflop_quality(cards)
    if kind == "NIT/FOLD_HEAVY":
        weight = 0.12 + 2.8 * q * q
    elif kind == "PASSIVE_CALLER":
        weight = 0.72 + 0.48 * q
    elif kind == "AGGRESSIVE_MANIAC":
        weight = 0.82 + 0.36 * q
    else:
        weight = 0.38 + 1.10 * q
    # Showdown holdings reveal how strong the opponent's entered range really
    # is.  Use this only after enough observations to avoid overfitting noise.
    if profile.get("showdown_samples", 0) >= 6:
        center = profile.get("showdown_preflop_quality", 0.55)
        weight *= _clamp(1.0 + 1.25 * (q - center), 0.62, 1.48)
    return weight


def _estimate_equity(hole, board, profile, rng, samples):
    """Monte Carlo equity against a style-weighted five-card opponent range.

    Common random numbers are used for both players on each rollout. Sampling
    and scoring are bounded by `samples`; no unbounded rejection loop exists.
    """
    missing = 5 - len(board)
    if missing < 0 or missing > 2 or len(hole) != 5:
        return 0.50
    known = set(hole)
    known.update(board)
    deck = [c for c in range(36) if c not in known]
    take = 5 + missing
    if len(deck) < take:
        return 0.50
    table = _table()
    win = 0.0
    total = 0.0
    for _ in range(max(1, samples)):
        draw = rng.sample(deck, take)
        opp_hole = draw[:5]
        if missing:
            full_board = tuple(board) + tuple(draw[5:])
            board_masks = _fast_board_masks(full_board)
        else:
            full_board = board
            board_masks = _fast_board_masks(full_board)
        hero_score = _fast_omaha_score(hole, board_masks, table)
        opp_score = _fast_omaha_score(opp_hole, board_masks, table)
        weight = _opponent_range_weight(opp_hole, profile)
        if hero_score > opp_score:
            win += weight
        elif hero_score == opp_score:
            win += 0.5 * weight
        total += weight
    raw = win / total if total else 0.50
    # Shrink noisy low-sample estimates toward the heads-up prior.
    shrink = 0.82 if missing == 0 else 0.72
    return _clamp(0.50 + (raw - 0.50) * shrink, 0.02, 0.98)


def _draw_features(hole, board):
    """Return (flush_draw, nut_flush_blocker, straight_draw, combo_draw)."""
    hole_suits = [0] * 4
    board_suits = [0] * 4
    for c in hole:
        hole_suits[c & 3] += 1
    for c in board:
        board_suits[c & 3] += 1
    flush_draw = False
    nut_flush_blocker = False
    for s in range(4):
        if len(board) < 5 and hole_suits[s] >= 2 and board_suits[s] == 2:
            flush_draw = True
        # A river blocker is meaningful only when three board cards of that
        # suit allow an opponent to make a flush using exactly three board cards.
        blocker_board_count = 2 if len(board) < 5 else 3
        if board_suits[s] >= blocker_board_count and any(
                (c & 3) == s and (c >> 2) == 8 for c in hole):
            nut_flush_blocker = True

    # On each legal 3-card subset of the visible board, combine with each
    # 2-card hole subset. A river has no remaining draw, so skip this test.
    straight_draw = False
    if len(board) < 5:
        board_triples = _itertools.combinations(range(len(board)), 3)
        board_triples = tuple(board_triples)
        for hi, hj in _HOLE_PAIRS:
            hr0 = hole[hi] >> 2
            hr1 = hole[hj] >> 2
            if hr0 == hr1:
                continue
            for bi, bj, bk in board_triples:
                ranks = ((1 << hr0) | (1 << hr1) |
                         (1 << (board[bi] >> 2)) |
                         (1 << (board[bj] >> 2)) |
                         (1 << (board[bk] >> 2)))
                for window in _SHORT_DECK_STRAIGHTS:
                    if (ranks & window).bit_count() >= 4:
                        straight_draw = True
                        break
                if straight_draw:
                    break
            if straight_draw:
                break
    return flush_draw, nut_flush_blocker, straight_draw, (flush_draw and straight_draw)


def _river_nut_rank(hole, board):
    """Exact best/second-best possible opponent pair strength on the river.

    Any two unseen cards can be the two hole cards used by an opponent's PLO
    hand; the other three hole cards can be filled from the remaining deck.
    Therefore 465 legal unseen pairs suffice to establish the maximum possible
    opponent hand score on this fixed board.
    Returns 2 (top/nut score), 1 (at least second distinct score), or 0.
    """
    blocked = set(hole)
    blocked.update(board)
    remaining = [c for c in range(36) if c not in blocked]
    board_masks = _fast_board_masks(board)
    table = _table()
    best = -1
    second = -1
    for i in range(len(remaining) - 1):
        p0 = 1 << remaining[i]
        for j in range(i + 1, len(remaining)):
            pair_mask = p0 | (1 << remaining[j])
            pair_best = 0
            for bm in board_masks:
                v = table[pair_mask | bm]
                if v > pair_best:
                    pair_best = v
            if pair_best > best:
                second, best = best, pair_best
            elif second < pair_best < best:
                second = pair_best
    hero_score = _fast_omaha_score(hole, board_masks, table)
    if hero_score >= best:
        return 2
    if hero_score >= second:
        return 1
    return 0


def _opponent_profile(memory):
    hands = max(0, int(memory.get("opp_preflop_hands", 0)))
    vpip_n = int(memory.get("opp_vpip_hands", 0))
    pfr_n = int(memory.get("opp_pfr_hands", 0))
    vpip = vpip_n / hands if hands >= 8 else 0.45
    pfr = pfr_n / hands if hands >= 8 else 0.22
    raises = int(memory.get("opp_postflop_raises", 0))
    calls = int(memory.get("opp_postflop_calls", 0))
    af = raises / max(1, raises + calls) if raises + calls >= 6 else 0.30
    faced = int(memory.get("opp_raise_opportunities", 0))
    folds = int(memory.get("opp_folds_to_raise", 0))
    fold_rate = folds / faced if faced >= 5 else 0.50

    kind = "UNKNOWN"
    if hands >= 18:
        if (af >= 0.48 and (pfr >= 0.22 or vpip >= 0.58)):
            kind = "AGGRESSIVE_MANIAC"
        elif vpip >= 0.56 and pfr <= 0.24 and af <= 0.34:
            kind = "PASSIVE_CALLER"
        elif ((vpip <= 0.27 and pfr <= 0.18) or
              (faced >= 7 and fold_rate >= 0.72)):
            kind = "NIT/FOLD_HEAVY"
        elif faced >= 9 and fold_rate >= 0.76:
            kind = "NIT/FOLD_HEAVY"
        else:
            kind = "BALANCED"
    elif faced >= 7 and fold_rate >= 0.78:
        kind = "NIT/FOLD_HEAVY"
    showdown_samples = int(memory.get("opp_showdown_quality_count", 0))
    showdown_quality = (float(memory.get("opp_showdown_quality_sum", 0.0)) /
                        showdown_samples if showdown_samples else 0.55)
    return {"kind": kind, "hands": hands, "vpip": vpip, "pfr": pfr,
            "aggression": af, "fold_to_raise": fold_rate,
            "showdowns": int(memory.get("opp_showdowns", 0)),
            "showdown_samples": showdown_samples,
            "showdown_preflop_quality": showdown_quality}


def _process_action_history(obs, memory):
    """Increment opponent counters exactly once as the visible history grows."""
    hand_index = int(obs.hand_index)
    if memory.get("_history_hand") != hand_index:
        memory["_history_hand"] = hand_index
        memory["_history_seen"] = 0
        memory["_opp_vpip_this_hand"] = False
        memory["_opp_pfr_this_hand"] = False
        memory["_raise_waiting"] = False
    history = obs.action_history
    start = min(int(memory.get("_history_seen", 0)), len(history))
    for rec in history[start:]:
        if rec.is_me:
            if rec.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["_raise_waiting"] = True
            continue

        if memory.get("_raise_waiting"):
            memory["opp_raise_opportunities"] = int(memory.get("opp_raise_opportunities", 0)) + 1
            if rec.action == Action.FOLD:
                memory["opp_folds_to_raise"] = int(memory.get("opp_folds_to_raise", 0)) + 1
            memory["_raise_waiting"] = False

        if rec.street == Street.PREFLOP:
            if rec.action != Action.FOLD and int(rec.amount) > 0 and not memory.get("_opp_vpip_this_hand"):
                memory["opp_vpip_hands"] = int(memory.get("opp_vpip_hands", 0)) + 1
                memory["_opp_vpip_this_hand"] = True
            if rec.action in (Action.RAISE_HALF, Action.RAISE_POT) and not memory.get("_opp_pfr_this_hand"):
                memory["opp_pfr_hands"] = int(memory.get("opp_pfr_hands", 0)) + 1
                memory["_opp_pfr_this_hand"] = True
        else:
            if rec.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_postflop_raises"] = int(memory.get("opp_postflop_raises", 0)) + 1
            elif rec.action == Action.CHECK_CALL and int(rec.amount) > 0:
                memory["opp_postflop_calls"] = int(memory.get("opp_postflop_calls", 0)) + 1
    memory["_history_seen"] = len(history)


def _adaptive_samples(obs, memory, street):
    """Keep rollout work inside a conservative share of the half time bank."""
    bank = float(memory.get("time_bank", 8.0))
    used = max(0.0, float(memory.get("think_used", 0.0)))
    remaining = bank - used
    # Preserve time for framework/JSON overhead and late-hand uncertainty.
    hands_left = max(1, int(memory.get("num_hands", 1000)) - int(obs.hand_index))
    expected_calls = hands_left * 2.55
    remaining -= 0.90 + expected_calls * 0.00010
    if remaining < 2.0:
        return 0
    per_call = remaining / max(1.0, expected_calls)
    if per_call < 0.00085:
        return 0
    if per_call < 0.00145:
        return 8 if street == Street.FLOP else 10 if street == Street.TURN else 10
    if street == Street.FLOP:
        return 18
    if street == Street.TURN:
        return 22
    return 20


def _pot_odds(obs):
    call = max(0, int(obs.to_call))
    pot = max(0, int(obs.pot))
    return call / max(1, pot + call)


def _choose_raise(obs, prefer_pot=False):
    legal = obs.legal_actions
    if prefer_pot and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    if Action.RAISE_POT in legal:
        return Action.RAISE_POT
    return None


def _preflop_action(obs, memory, profile):
    legal = obs.legal_actions
    quality = _preflop_quality(obs.hole_cards)
    equity = _preflop_equity(obs.hole_cards)
    raises = int(obs.raises_this_street)
    if profile["kind"] == "AGGRESSIVE_MANIAC":
        margin = 0.025
    elif profile["kind"] == "NIT/FOLD_HEAVY":
        margin = 0.070
    else:
        margin = 0.045
    if obs.to_call > 0:
        threshold = _pot_odds(obs) + margin + 0.018 * min(2, raises)
        if obs.my_stack <= 2 * obs.big_blind:
            threshold -= 0.02
        if equity < threshold:
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        # Raise only the very best structures over a voluntary raise.
        if equity >= 0.69 and quality >= 0.76:
            r = _choose_raise(obs, prefer_pot=(equity >= 0.71))
            if r is not None:
                return r
        return Action.CHECK_CALL

    # No price: enter pots cheaply, but raise premiums and steal when an
    # established opponent profile shows unusually frequent preflop folds.
    raise_cut = 0.65
    if profile["kind"] == "AGGRESSIVE_MANIAC":
        raise_cut = 0.68
    elif profile["kind"] == "PASSIVE_CALLER":
        raise_cut = 0.63
    elif profile["kind"] == "NIT/FOLD_HEAVY":
        raise_cut = 0.58
    if equity >= raise_cut and quality >= 0.68:
        r = _choose_raise(obs, prefer_pot=(equity >= 0.70))
        if r is not None:
            return r
    return Action.CHECK_CALL


def _postflop_action(obs, memory, profile):
    legal = obs.legal_actions
    street = obs.street
    samples = _adaptive_samples(obs, memory, street)
    flush_draw, nut_blocker, straight_draw, combo_draw = _draw_features(obs.hole_cards, obs.board)

    # Exact river hand class and exact top/second-nut test are inexpensive;
    # the latter is skipped in emergency mode to protect the time bank.
    nut_rank = 0
    made_category = -1
    if street == Street.RIVER:
        score = omaha_score(obs.hole_cards, obs.board)
        made_category = score >> 20
        if samples > 0:
            nut_rank = _river_nut_rank(obs.hole_cards, obs.board)
    if samples > 0:
        equity = _estimate_equity(obs.hole_cards, obs.board, profile,
                                  memory["rng"], samples)
    else:
        # Emergency fallback: cheap structure/made-hand rules only.
        if street == Street.RIVER:
            equity = 0.66 if nut_rank == 2 else 0.56 if made_category >= 6 else 0.49 if made_category >= 4 else 0.38
        else:
            equity = _preflop_equity(obs.hole_cards)
            if combo_draw:
                equity = min(0.70, equity + 0.10)
            elif flush_draw or straight_draw:
                equity = min(0.65, equity + 0.055)

    pot = max(1, int(obs.pot))
    to_call = max(0, int(obs.to_call))
    odds = _pot_odds(obs)
    position_bonus = 0.018 if obs.is_button else 0.0  # button acts last postflop
    risk = 0.025 if (obs.my_stack + obs.opp_stack) > (2 * pot) else 0.010
    if flush_draw or straight_draw:
        risk *= 0.45
    if profile["kind"] == "AGGRESSIVE_MANIAC":
        profile_call_adj = -0.055
    elif profile["kind"] == "NIT/FOLD_HEAVY":
        profile_call_adj = 0.045
    elif profile["kind"] == "PASSIVE_CALLER":
        profile_call_adj = 0.012
    else:
        profile_call_adj = 0.0
    if profile["showdown_samples"] >= 8:
        if profile["showdown_preflop_quality"] >= 0.68:
            profile_call_adj += 0.018
        elif profile["showdown_preflop_quality"] <= 0.48:
            profile_call_adj -= 0.018

    call_threshold = odds + 0.035 + risk + profile_call_adj - position_bonus
    # An extremely large bet into a capped pot deserves more respect unless
    # we have a strong made hand, a nut draw, or credible maniac evidence.
    if to_call >= pot:
        call_threshold += 0.045
    if to_call >= 2 * pot:
        call_threshold += 0.055
    river_bluff_catcher = (street == Street.RIVER and nut_rank == 0 and
                           made_category <= 2)
    if nut_rank == 2:
        call_threshold = min(call_threshold, odds + 0.015)
    elif nut_rank == 1:
        # Second-nut hands are strong bluff-catchers, but retain some caution
        # because the unique top hand is still possible.
        call_threshold = min(call_threshold, odds + 0.045)
    elif river_bluff_catcher:
        call_threshold += -0.015 if profile["kind"] == "AGGRESSIVE_MANIAC" else 0.025
    elif made_category >= 7:
        call_threshold = min(call_threshold, odds + 0.025)
    elif made_category == 6 and street == Street.RIVER:
        call_threshold = min(call_threshold, odds + 0.055)

    # Value raise / semi-bluff thresholds. The bot does not blindly pot-raise
    # all draws: it needs equity, a nut blocker, or an opponent who overfolds.
    raise_value_threshold = 0.755
    if profile["kind"] == "AGGRESSIVE_MANIAC":
        raise_value_threshold = 0.715
    elif profile["kind"] == "PASSIVE_CALLER":
        raise_value_threshold = 0.790
    elif profile["kind"] == "NIT/FOLD_HEAVY":
        raise_value_threshold = 0.735

    if to_call == 0:
        if equity >= raise_value_threshold:
            r = _choose_raise(obs, prefer_pot=(equity >= 0.84 or nut_rank == 2))
            if r is not None:
                return r
        if (profile["kind"] == "NIT/FOLD_HEAVY" and nut_blocker and
                equity >= 0.49 and street != Street.RIVER):
            r = _choose_raise(obs, prefer_pot=False)
            if r is not None:
                return r
        if combo_draw and nut_blocker and equity >= 0.57:
            r = _choose_raise(obs, prefer_pot=(equity >= 0.65))
            if r is not None:
                return r
        return Action.CHECK_CALL

    # Strong made hands can raise for value; combination draws may semi-bluff
    # against opponents with a verified high fold-to-raise rate.
    if equity >= raise_value_threshold + 0.045:
        r = _choose_raise(obs, prefer_pot=(equity >= 0.86 or nut_rank == 2))
        if r is not None:
            return r
    if (profile["kind"] == "NIT/FOLD_HEAVY" and nut_blocker and
            equity >= 0.55 and street != Street.RIVER):
        r = _choose_raise(obs, prefer_pot=False)
        if r is not None:
            return r
    if combo_draw and nut_blocker and equity >= 0.64 and profile["kind"] != "PASSIVE_CALLER":
        r = _choose_raise(obs, prefer_pot=(equity >= 0.72))
        if r is not None:
            return r

    if equity >= call_threshold:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def new_half(config: AgentConfig, memory: dict) -> None:
    """Initialize per-half opponent statistics, RNG, and a conservative time model."""
    try:
        try:
            bank = float(getattr(config, "time_bank_seconds", 8.0))
            num_hands = int(getattr(config, "num_hands", 1000))
        except Exception:
            bank, num_hands = 8.0, 1000
        memory.clear()
        memory.update({
            "hands_played": 0,
            "num_hands": max(1, num_hands),
            "time_bank": max(1.0, bank),
            "think_used": 0.0,
            "rng": _random.Random(),
            "opp_preflop_hands": 0,
            "opp_vpip_hands": 0,
            "opp_pfr_hands": 0,
            "opp_postflop_raises": 0,
            "opp_postflop_calls": 0,
            "opp_raise_opportunities": 0,
            "opp_folds_to_raise": 0,
            "opp_showdowns": 0,
            "opp_showdown_categories": [0] * 9,
            "opp_showdown_quality_sum": 0.0,
            "opp_showdown_quality_count": 0,
            "_history_hand": None,
            "_history_seen": 0,
            "_opp_vpip_this_hand": False,
            "_opp_pfr_this_hand": False,
            "_raise_waiting": False,
            "_last_ended_hand": -1,
        })
    except Exception:
        # The route should never see an exception from the agent callback.
        try:
            memory.clear()
            memory["hands_played"] = 0
            memory["time_bank"] = 8.0
            memory["think_used"] = 0.0
            memory["rng"] = _random.Random()
        except Exception:
            pass


def choose_action(obs: Observation, memory: dict) -> Action:
    """Choose a legal PLO5 action with bounded equity sampling and profiling."""
    started = _time.perf_counter()
    try:
        if not isinstance(memory, dict):
            return _fallback_action(obs)
        if "rng" not in memory:
            # Defensive initialization if called without /new-half in a test harness.
            new_half(None, memory)
        _process_action_history(obs, memory)
        profile = _opponent_profile(memory)
        if obs.street == Street.PREFLOP:
            action = _preflop_action(obs, memory, profile)
        else:
            action = _postflop_action(obs, memory, profile)
        legal = obs.legal_actions
        if action not in legal:
            action = _fallback_action(obs)
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["_raise_waiting"] = True
        return action
    except Exception:
        # A malformed/edge observation must degrade to a legal check/fold.
        return _fallback_action(obs)
    finally:
        try:
            memory["think_used"] = float(memory.get("think_used", 0.0)) + (_time.perf_counter() - started)
        except Exception:
            pass


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Update delayed opponent statistics and record revealed showdown holdings."""
    started = _time.perf_counter()
    try:
        if not isinstance(memory, dict):
            return
        hand_index = int(getattr(result, "hand_index", -1))
        if hand_index == int(memory.get("_last_ended_hand", -2)):
            return

        # A non-showdown win immediately after our raise means the opponent
        # folded to that raise; the fold itself is not exposed in HandResult.
        if (bool(getattr(result, "winner", "" ) == "me") and
                not bool(getattr(result, "went_to_showdown", False)) and
                bool(memory.get("_raise_waiting", False))):
            memory["opp_raise_opportunities"] = int(memory.get("opp_raise_opportunities", 0)) + 1
            memory["opp_folds_to_raise"] = int(memory.get("opp_folds_to_raise", 0)) + 1
        memory["_raise_waiting"] = False
        memory["opp_preflop_hands"] = int(memory.get("opp_preflop_hands", 0)) + 1

        opp_hole = getattr(result, "opp_hole", None)
        board = getattr(result, "board", ())
        if bool(getattr(result, "went_to_showdown", False)) and opp_hole is not None and len(board) == 5:
            try:
                category = omaha_score(opp_hole, board) >> 20
                cats = memory.get("opp_showdown_categories")
                if not isinstance(cats, list) or len(cats) != 9:
                    cats = [0] * 9
                    memory["opp_showdown_categories"] = cats
                if 0 <= category < 9:
                    cats[category] += 1
                    memory["opp_showdowns"] = int(memory.get("opp_showdowns", 0)) + 1
                    memory["opp_showdown_quality_sum"] = (
                        float(memory.get("opp_showdown_quality_sum", 0.0)) +
                        _preflop_quality(opp_hole))
                    memory["opp_showdown_quality_count"] = (
                        int(memory.get("opp_showdown_quality_count", 0)) + 1)
            except Exception:
                # A damaged showdown record should not affect future decisions.
                pass

        memory["hands_played"] = int(memory.get("hands_played", 0)) + 1
        memory["_last_ended_hand"] = hand_index
    except Exception:
        # Callback failures count as errors in the tournament; never propagate.
        pass
    finally:
        try:
            memory["think_used"] = float(memory.get("think_used", 0.0)) + (_time.perf_counter() - started)
        except Exception:
            pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
