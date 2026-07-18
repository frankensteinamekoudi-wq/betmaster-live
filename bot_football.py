import requests
import json
import os
from datetime import datetime, timedelta
import time
from math import exp, factorial

# Configuration
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY')
RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"

LEAGUES = {
    "PL": 39,    # Premier League
    "PD": 140,   # La Liga
    "BL1": 78,   # Bundesliga
    "SA": 135,   # Serie A
    "FL1": 61,   # Ligue 1
    "CL": 2      # Champions League
}

RATE_LIMIT = 8  # 8 req/min pour rester sous les 10 req/min autorisées
last_request_time = 0

def rate_limit():
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < (60 / RATE_LIMIT):
        time.sleep((60 / RATE_LIMIT) - elapsed)
    last_request_time = time.time()

def get_league_standings(league_id):
    """Récupère le classement avec stats réelles"""
    rate_limit()
    url = "https://api-football-v1.p.rapidapi.com/v3/standings"
    querystring = {"league": league_id, "season": "2026"}
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        print(f"📊 Standings {league_id}: status {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('response') and data['response'][0].get('league', {}).get('standings'):
                standings = data['response'][0]['league']['standings'][0]
                teams = []
                for team in standings:
                    teams.append({
                        'name': team['team']['name'],
                        'played': team['all']['played'],
                        'goals_for': team['all']['goals']['for'],
                        'goals_against': team['all']['goals']['against'],
                        'points': team['points']
                    })
                return teams
        return []
    except Exception as e:
        print(f"❌ Erreur standings {league_id}: {e}")
        return []

def get_fixtures(league_id, days_ahead=7):
    """Récupère les matchs programmés des 7 prochains jours"""
    rate_limit()
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {
        "league": league_id,
        "season": "2026",
        "from": today,
        "to": next_week
    }
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        print(f"📊 Fixtures {league_id}: status {response.status_code}, {len(response.json().get('response', []))} matchs")
        
        if response.status_code == 200:
            data = response.json()
            fixtures = []
            for fixture in data.get('response', []):
                fixtures.append({
                    'home': fixture['teams']['home']['name'],
                    'away': fixture['teams']['away']['name'],
                    'date': fixture['fixture']['date'],
                    'id': fixture['fixture']['id']
                })
            return fixtures
        return []
    except Exception as e:
        print(f"❌ Erreur fixtures {league_id}: {e}")
        return []

def calculate_team_strength(teams):
    """Calcule force offensive/défensive"""
    if not teams:
        return {}, {}
    
    total_goals_for = sum(t['goals_for'] for t in teams)
    total_goals_against = sum(t['goals_against'] for t in teams)
    total_played = sum(t['played'] for t in teams)
    
    if total_played == 0:
        return {}, {}
    
    avg_for = total_goals_for / total_played
    avg_against = total_goals_against / total_played
    
    attack = {}
    defense = {}
    for team in teams:
        played = team['played']
        if played > 0:
            attack[team['name']] = (team['goals_for'] / played) / avg_for
            defense[team['name']] = avg_against / (team['goals_against'] / played)
        else:
            attack[team['name']] = 1.0
            defense[team['name']] = 1.0
    
    return attack, defense

def poisson_prob(goals, lamb):
    """Probabilité de Poisson pour un nombre de buts"""
    if lamb <= 0:
        return 1.0 if goals == 0 else 0.0
    return (exp(-lamb) * (lamb ** goals)) / factorial(goals)

def predict_match(home_team, away_team, attack, defense, home_advantage=1.35):
    """Prédiction Poisson"""
    home_attack = attack.get(home_team, 1.0) * home_advantage
    home_defense = defense.get(home_team, 1.0)
    away_attack = attack.get(away_team, 1.0)
    away_defense = defense.get(away_team, 1.0)
    
    # Buts attendus
    home_goals = home_attack * away_defense
    away_goals = away_attack * home_defense
    
    # Probabilités Poisson 0-5 buts
    home_probs = [poisson_prob(i, home_goals) for i in range(6)]
    away_probs = [poisson_prob(i, away_goals) for i in range(6)]
    
    # Probabilité 1 (victoire domicile)
    prob_1 = sum(home_probs[i] * sum(away_probs[:i]) for i in range(1, 6))
    # Probabilité X (nul)
    prob_X = sum(home_probs[i] * away_probs[i] for i in range(6))
    # Probabilité 2 (victoire extérieur)
    prob_2 = sum(home_probs[i] * sum(away_probs[i+1:]) for i in range(5))
    
    # Normalisation
    total = prob_1 + prob_X + prob_2
    if total > 0:
        prob_1 /= total
        prob_X /= total
        prob_2 /= total
    
    # Cotes justes
    fair_odds_1 = 1 / prob_1 if prob_1 > 0 else 100
    fair_odds_X = 1 / prob_X if prob_X > 0 else 100
    fair_odds_2 = 1 / prob_2 if prob_2 > 0 else 100
    
    # Cotes avec marge bookmaker 7.5%
    margin = 0.075
    odd_1 = fair_odds_1 * (1 + margin)
    odd_X = fair_odds_X * (1 + margin)
    odd_2 = fair_odds_2 * (1 + margin)
    
    # Kelly et EV
    kelly_1 = (prob_1 - (1 / odd_1)) / (1 - (1 / odd_1)) if odd_1 > 1 else 0
    kelly_X = (prob_X - (1 / odd_X)) / (1 - (1 / odd_X)) if odd_X > 1 else 0
    kelly_2 = (prob_2 - (1 / odd_2)) / (1 - (1 / odd_2)) if odd_2 > 1 else 0
    
    ev_1 = (prob_1 * odd_1) - 1
    ev_X = (prob_X * odd_X) - 1
    ev_2 = (prob_2 * odd_2) - 1
    
    return {
        'home_goals_expect': round(home_goals, 3),
        'away_goals_expect': round(away_goals, 3),
        'prob_1': round(prob_1, 4),
        'prob_X': round(prob_X, 4),
        'prob_2': round(prob_2, 4),
        'fair_odds_1': round(fair_odds_1, 2),
        'fair_odds_X': round(fair_odds_X, 2),
        'fair_odds_2': round(fair_odds_2, 2),
        'odd_1': round(odd_1, 2),
        'odd_X': round(odd_X, 2),
        'odd_2': round(odd_2, 2),
        'kelly_1': round(kelly_1, 4),
        'kelly_X': round(kelly_X, 4),
        'kelly_2': round(kelly_2, 4),
        'ev_1': round(ev_1, 4),
        'ev_X': round(ev_X, 4),
        'ev_2': round(ev_2, 4),
        'predicted_score': f"{round(home_goals)}-{round(away_goals)}"
    }

def main():
    print("🚀 BETMASTER AI - Récupération des données...")
    print(f"🔑 Clé API: {RAPIDAPI_KEY[:10]}...")
    
    all_data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "competitions": {}
    }
    
    for league_name, league_id in LEAGUES.items():
        print(f"\n📊 Traitement de {league_name} (ID: {league_id})...")
        
        # Récupère le classement
        teams = get_league_standings(league_id)
        if not teams:
            print(f"⚠️ Aucun classement pour {league_name}")
            all_data["competitions"][league_name] = {"matches": []}
            continue
        
        print(f"✅ {league_name}: {len(teams)} équipes récupérées")
        
        # Calcule les forces
        attack, defense = calculate_team_strength(teams)
        
        # Récupère les fixtures
        fixtures = get_fixtures(league_id)
        if not fixtures:
            print(f"⚠️ Aucun match programmé pour {league_name} dans les 7 prochains jours")
            all_data["competitions"][league_name] = {"matches": []}
            continue
        
        # Prédit chaque match
        matches = []
        for fixture in fixtures:
            pred = predict_match(fixture['home'], fixture['away'], attack, defense)
            pred.update({
                'home_team': fixture['home'],
                'away_team': fixture['away'],
                'date': fixture['date'],
                'match_id': fixture['id']
            })
            matches.append(pred)
        
        all_data["competitions"][league_name] = {
            "standings": teams,
            "matches": matches
        }
        print(f"✅ {league_name}: {len(matches)} matchs prédits")
    
    # Sauvegarde
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Données sauvegardées dans data.json")
    print(f"📊 {sum(len(c.get('matches', [])) for c in all_data['competitions'].values())} matchs au total")

if __name__ == "__main__":
    main()
