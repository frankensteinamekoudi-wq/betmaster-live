import requests
import time
import json
import os
import math

# ================================================================
# CONFIGURATION
# ================================================================

FOOTBALL_API_KEY = os.environ["FOOTBALL_API_KEY"]
BASE_URL = "https://api.football-data.org/v4"

# Compétitions couvertes par le plan gratuit
COMPETITIONS = ["PL", "PD", "BL1", "SA", "FL1", "CL"]

DAYS_AHEAD = 7  # matchs programmés dans les N prochains jours
DATA_FILE = "data.json"

HOME_ADVANTAGE = 1.35  # multiplicateur avantage terrain (calibré empiriquement, ~35% de buts en plus à domicile)
BOOKMAKER_MARGIN = 1.075  # marge de 7.5%, cohérent avec BlackScanner

# ================================================================
# LIMITEUR DE DÉBIT : plan gratuit football-data.org = 10 req/min
# ================================================================

CALL_TIMESTAMPS = []
def wait_for_rate_limit():
    global CALL_TIMESTAMPS
    while True:
        now = time.time()
        CALL_TIMESTAMPS = [t for t in CALL_TIMESTAMPS if now - t < 60]
        if len(CALL_TIMESTAMPS) < 8:  # marge de sécurité sous 10
            CALL_TIMESTAMPS.append(now)
            return
        sleep_time = 60 - (now - CALL_TIMESTAMPS[0]) + 0.5
        print(f"⏳ Rate limit : pause {sleep_time:.0f}s")
        time.sleep(max(sleep_time, 1))

def api_get(path):
    wait_for_rate_limit()
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, headers={"X-Auth-Token": FOOTBALL_API_KEY}, timeout=15)
        if resp.status_code == 429:
            print("⏳ 429 reçu, pause 65s")
            time.sleep(65)
            return api_get(path)
        if resp.status_code != 200:
            print(f"Erreur HTTP {resp.status_code} sur {path}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Erreur fetch {path}: {e}")
        return None

# ================================================================
# STATISTIQUES RÉELLES : classement -> force d'attaque/défense par équipe
# (méthode inspirée de Dixon-Coles simplifiée, sans décroissance temporelle)
# ================================================================

def compute_team_strengths(standings_data):
    """A partir du classement réel (buts marqués/encaissés), calcule la force
    d'attaque et de défense de chaque équipe relative à la moyenne de la ligue."""
    if not standings_data or "standings" not in standings_data:
        return None, None, None

    table = None
    for s in standings_data["standings"]:
        if s.get("type") == "TOTAL":
            table = s.get("table", [])
            break
    if not table:
        return None, None, None

    total_games = sum(t["playedGames"] for t in table if t["playedGames"] > 0)
    if total_games == 0:
        return None, None, None

    total_goals_for = sum(t["goalsFor"] for t in table)
    league_avg_goals_per_team_game = total_goals_for / total_games if total_games else 1.3

    strengths = {}
    for t in table:
        played = t["playedGames"]
        if played == 0:
            continue
        team_id = t["team"]["id"]
        avg_scored = t["goalsFor"] / played
        avg_conceded = t["goalsAgainst"] / played
        strengths[team_id] = {
            "name": t["team"]["name"],
            "attack": avg_scored / league_avg_goals_per_team_game if league_avg_goals_per_team_game else 1.0,
            "defense": avg_conceded / league_avg_goals_per_team_game if league_avg_goals_per_team_game else 1.0,
            "played": played,
            "won": t["won"], "draw": t["draw"], "lost": t["lost"],
            "form": t.get("form", ""),
        }
    return strengths, league_avg_goals_per_team_game, total_games

# ================================================================
# MODÈLE POISSON (identique en esprit à BlackScanner, mais xG réels)
# ================================================================

def poisson_prob(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def predict_score_matrix(xg_h, xg_a, max_goals=6):
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[(h, a)] = poisson_prob(xg_h, h) * poisson_prob(xg_a, a)
    return matrix

def match_outcome_probs(matrix):
    p_home = sum(p for (h, a), p in matrix.items() if h > a)
    p_draw = sum(p for (h, a), p in matrix.items() if h == a)
    p_away = sum(p for (h, a), p in matrix.items() if h < a)
    total = p_home + p_draw + p_away
    return (p_home / total * 100, p_draw / total * 100, p_away / total * 100) if total else (33.3, 33.3, 33.3)

def over_under_prob(matrix, line):
    p_under = sum(p for (h, a), p in matrix.items() if (h + a) <= line)
    total = sum(matrix.values())
    p_under_pct = (p_under / total * 100) if total else 50
    return {"over": round(100 - p_under_pct, 1), "under": round(p_under_pct, 1)}

def btts_prob(matrix):
    p_btts = sum(p for (h, a), p in matrix.items() if h > 0 and a > 0)
    total = sum(matrix.values())
    return round((p_btts / total * 100), 1) if total else 50

def best_score(matrix):
    best = max(matrix.items(), key=lambda kv: kv[1])
    (h, a), p = best
    total = sum(matrix.values())
    return f"{h}-{a}", round(p / total * 100, 1)

def fair_odd(prob_pct):
    p = max(prob_pct, 1) / 100
    return round(max(1.03, BOOKMAKER_MARGIN / p), 2)

def kelly_criterion(prob_pct, odd):
    b = odd - 1
    p = prob_pct / 100
    q = 1 - p
    if b <= 0:
        return 0
    kelly = ((b * p) - q) / b
    return round(max(0, kelly * 100), 1)

def is_value_bet(prob_pct, odd):
    implied = 100 / odd
    return prob_pct > implied * 1.05

def expected_value(prob_pct, odd, stake=1000):
    p = prob_pct / 100
    gain = p * (odd - 1) * stake
    loss = (1 - p) * stake
    return round(gain - loss)

# ================================================================
# CALCUL COMPLET PAR MATCH (statistiques réelles + Poisson)
# ================================================================

def analyze_match(match, strengths, league_avg):
    home_id = match["homeTeam"]["id"]
    away_id = match["awayTeam"]["id"]
    home_s = strengths.get(home_id)
    away_s = strengths.get(away_id)

    if not home_s or not away_s:
        return None  # équipe promue/relégable sans historique suffisant cette saison

    xg_home = league_avg * home_s["attack"] * away_s["defense"] * HOME_ADVANTAGE
    xg_away = league_avg * away_s["attack"] * home_s["defense"]
    xg_home = max(0.2, min(xg_home, 4.5))
    xg_away = max(0.2, min(xg_away, 4.5))

    matrix = predict_score_matrix(xg_home, xg_away)
    prob_h, prob_d, prob_a = match_outcome_probs(matrix)
    ou25 = over_under_prob(matrix, 2)
    ou15 = over_under_prob(matrix, 1)
    ou35 = over_under_prob(matrix, 3)
    p_btts = btts_prob(matrix)
    score, score_prob = best_score(matrix)

    odd_h, odd_d, odd_a = fair_odd(prob_h), fair_odd(prob_d), fair_odd(prob_a)
    odd_o25, odd_u25 = fair_odd(ou25["over"]), fair_odd(ou25["under"])
    odd_btts = fair_odd(p_btts)
    odd_btts_n = fair_odd(100 - p_btts)

    candidates = [
        {"label": f"1 {home_s['name']}", "prob": prob_h, "odd": odd_h},
        {"label": f"2 {away_s['name']}", "prob": prob_a, "odd": odd_a},
        {"label": "Nul", "prob": prob_d, "odd": odd_d},
        {"label": "Over 2.5", "prob": ou25["over"], "odd": odd_o25},
        {"label": "BTTS Oui", "prob": p_btts, "odd": odd_btts},
    ]
    for c in candidates:
        c["ev"] = expected_value(c["prob"], c["odd"])
        c["value_bet"] = is_value_bet(c["prob"], c["odd"])
        c["kelly"] = kelly_criterion(c["prob"], c["odd"])

    best = max(candidates, key=lambda c: c["ev"])
    confidence = round(min(9.5, max(4.0, 5.0 + best["ev"] / 400 + (1.5 if best["value_bet"] else 0))), 1)
    risk = "Faible" if confidence > 7.5 else "Moyen" if confidence > 6 else "Élevé"

    return {
        "match_id": match["id"],
        "utc_date": match["utcDate"],
        "status": match["status"],
        "matchday": match.get("matchday"),
        "home_team": home_s["name"],
        "away_team": away_s["name"],
        "home_crest": match["homeTeam"].get("crest"),
        "away_crest": match["awayTeam"].get("crest"),
        "xg_home": round(xg_home, 2),
        "xg_away": round(xg_away, 2),
        "prob_home": round(prob_h, 1),
        "prob_draw": round(prob_d, 1),
        "prob_away": round(prob_a, 1),
        "odd_home": odd_h, "odd_draw": odd_d, "odd_away": odd_a,
        "over_under_2_5": ou25, "over_under_1_5": ou15, "over_under_3_5": ou35,
        "odd_over_2_5": odd_o25, "odd_under_2_5": odd_u25,
        "btts_prob": p_btts, "odd_btts": odd_btts, "odd_btts_no": odd_btts_n,
        "predicted_score": score, "predicted_score_prob": score_prob,
        "best_bet": best["label"], "best_bet_odd": best["odd"],
        "best_bet_ev": best["ev"], "best_bet_value": best["value_bet"], "best_bet_kelly": best["kelly"],
        "confidence": confidence, "risk": risk,
        "home_form": home_s.get("form", ""), "away_form": away_s.get("form", ""),
        "home_record": {"won": home_s["won"], "draw": home_s["draw"], "lost": home_s["lost"], "played": home_s["played"]},
        "away_record": {"won": away_s["won"], "draw": away_s["draw"], "lost": away_s["lost"], "played": away_s["played"]},
        "all_bets": candidates,
    }

# ================================================================
# EXÉCUTION PRINCIPALE
# ================================================================

def main():
    print(f"⚽ Run à {time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    export = {"updated_at": time.strftime('%Y-%m-%d %H:%M:%S') + " UTC", "competitions": {}}

    today = time.strftime('%Y-%m-%d')
    date_to = time.strftime('%Y-%m-%d', time.gmtime(time.time() + DAYS_AHEAD * 86400))

    for comp in COMPETITIONS:
        print(f"--- {comp} ---")

        standings = api_get(f"/competitions/{comp}/standings")
        strengths, league_avg, total_games = compute_team_strengths(standings)
        if not strengths:
            print(f"⚠️ Pas de classement exploitable pour {comp}, compétition sautée.")
            continue

        matches_data = api_get(f"/competitions/{comp}/matches?dateFrom={today}&dateTo={date_to}&status=SCHEDULED")
        if not matches_data or not matches_data.get("matches"):
            print(f"⚠️ Aucun match programmé pour {comp} sur les {DAYS_AHEAD} prochains jours.")
            export["competitions"][comp] = {"matches": []}
            continue

        analyzed = []
        for m in matches_data["matches"]:
            result = analyze_match(m, strengths, league_avg)
            if result:
                analyzed.append(result)

        analyzed.sort(key=lambda x: x["utc_date"])
        export["competitions"][comp] = {
            "matches": analyzed,
            "league_avg_goals": round(league_avg, 2) if league_avg else None,
        }
        print(f"✅ {len(analyzed)} matchs analysés pour {comp}")

    with open(DATA_FILE, "w") as f:
        json.dump(export, f, indent=2)
    print("Run terminé, data.json écrit.")

if __name__ == "__main__":
    main()
