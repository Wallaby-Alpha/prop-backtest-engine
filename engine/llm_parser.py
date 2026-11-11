import os
import json
import re

# Strict Schema of Allowed Strategy Configurations
STRATEGY_SCHEMA = {
    "Value Area Reclaim": {
        "start_time": {"type": "str", "default": "10:00", "pattern": r"^\d{2}:\d{2}$"},
        "end_time": {"type": "str", "default": "12:00", "pattern": r"^\d{2}:\d{2}$"},
        "target_mode": {"type": "str", "default": "opposite_va", "options": ["opposite_va", "poc", "scale_50_50"]},
        "ny_open_filter": {"type": "str", "default": "none", "options": ["none", "outside_range", "outside_va"]},
        "trend_filter": {"type": "str", "default": "none", "options": ["none", "daily_ema50", "daily_ema200"]},
        "stop_loss_buffer_atr": {"type": "float", "default": 0.5, "min": 0.0, "max": 3.0},
        "require_vol_surge": {"type": "bool", "default": False},
        "eod_exit_time": {"type": "str", "default": "15:55", "pattern": r"^\d{2}:\d{2}$"}
    },
    "Opening Range Breakout (ORB)": {
        "orb_start": {"type": "str", "default": "09:30", "pattern": r"^\d{2}:\d{2}$"},
        "orb_end": {"type": "str", "default": "10:00", "pattern": r"^\d{2}:\d{2}$"},
        "trade_end": {"type": "str", "default": "12:00", "pattern": r"^\d{2}:\d{2}$"},
        "r_target_mult": {"type": "float", "default": 2.0, "min": 1.0, "max": 5.0},
        "eod_exit_time": {"type": "str", "default": "15:55", "pattern": r"^\d{2}:\d{2}$"}
    }
}

ALLOWED_ASSETS = ["MNQ", "NQ", "BTC"]
ALLOWED_TIMEFRAMES = ["15m", "5m"]
ALLOWED_PROPS = ["Topstep $50k", "Topstep $100k", "Topstep $150k", "Apex $50k", "Static Drawdown $50k"]

SYSTEM_PROMPT = """You are an AI Trading Setup Compiler for an institutional backtesting and prop firm engine.
Your sole job is to translate user natural language strategy descriptions into a structured JSON parameter payload.

CRITICAL SECURITY & METHODOLOGY RULES:
1. You can ONLY modify trading setup parameters (entry window, filters, targets, asset, contracts, prop preset, risk scalers).
2. You CANNOT change backtesting execution methodology, fill logic, slippage, commissions, or prop firm simulation rules.
3. You must output VALID JSON only matching the schema below.

Allowed Strategies:
- "Value Area Reclaim"
- "Opening Range Breakout (ORB)"

Allowed Assets: ["MNQ", "NQ", "BTC"]
Allowed Timeframes: ["15m", "5m"]
Allowed Prop Presets: ["Topstep $50k", "Topstep $100k", "Topstep $150k", "Apex $50k", "Static Drawdown $50k"]

Value Area Reclaim Parameters:
- start_time: "HH:MM" (e.g. "10:00")
- end_time: "HH:MM" (e.g. "12:00")
- target_mode: "opposite_va" | "poc" | "scale_50_50"
- ny_open_filter: "none" | "outside_range" | "outside_va"
- trend_filter: "none" | "daily_ema50" | "daily_ema200"
- stop_loss_buffer_atr: float between 0.0 and 3.0 (default 0.5)
- require_vol_surge: boolean (default false)
- eod_exit_time: "HH:MM" (default "15:55")

ORB Parameters:
- orb_start: "09:30"
- orb_end: "10:00"
- trade_end: "12:00"
- r_target_mult: float between 1.0 and 5.0
- eod_exit_time: "15:55"

JSON Output Format:
{
  "strategy": "Value Area Reclaim",
  "asset": "MNQ",
  "timeframe": "15m",
  "contracts": 2,
  "prop_preset": "Topstep $50k",
  "eval_risk": 1.0,
  "funded_risk": 0.5,
  "params": { ... },
  "explanation": "Brief 1-2 sentence human summary of the configured setup"
}
"""

def parse_with_rule_fallback(user_text):
    """
    Fallback deterministic parser using pattern matching if no LLM API key is configured.
    """
    txt = user_text.lower()
    
    # Strategy
    strat = "Value Area Reclaim"
    if "orb" in txt or "opening range" in txt:
        strat = "Opening Range Breakout (ORB)"
        
    # Asset
    asset = "MNQ"
    if "btc" in txt or "bitcoin" in txt:
        asset = "BTC"
    elif " nq " in txt or "mini nq" in txt:
        asset = "NQ"
        
    # Timeframe
    tf = "15m"
    if "5m" in txt or "5 min" in txt:
        tf = "5m"
        
    # Prop Preset
    prop = "Topstep $50k"
    if "100k" in txt:
        prop = "Topstep $100k"
    elif "150k" in txt:
        prop = "Topstep $150k"
    elif "apex" in txt:
        prop = "Apex $50k"
    elif "static" in txt:
        prop = "Static Drawdown $50k"
        
    params = {}
    if strat == "Value Area Reclaim":
        # Times
        s_time = "10:00"
        e_time = "12:00"
        time_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:to|-|and)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", txt)
        if time_match:
            # simple normalization
            pass
            
        # Target mode
        target_mode = "opposite_va"
        if "poc" in txt and "opposite" not in txt:
            target_mode = "poc"
        elif "50/50" in txt or "scale" in txt:
            target_mode = "scale_50_50"
            
        # NY Open filter
        ny_filter = "none"
        if "outside" in txt and ("range" in txt or "yesterday's range" in txt or "prior range" in txt):
            ny_filter = "outside_range"
        elif "outside" in txt and ("value" in txt or "va" in txt):
            ny_filter = "outside_va"
            
        # Trend filter
        trend = "none"
        if "200" in txt and ("ema" in txt or "ma" in txt or "d ema" in txt):
            trend = "daily_ema200"
        elif "50" in txt and ("ema" in txt or "ma" in txt or "d ema" in txt):
            trend = "daily_ema50"
            
        params = {
            "start_time": s_time,
            "end_time": e_time,
            "target_mode": target_mode,
            "ny_open_filter": ny_filter,
            "trend_filter": trend,
            "stop_loss_buffer_atr": 0.5,
            "require_vol_surge": "volume" in txt and "surge" in txt,
            "eod_exit_time": "15:55"
        }
    else:
        params = {
            "orb_start": "09:30",
            "orb_end": "10:00",
            "trade_end": "12:00",
            "r_target_mult": 2.0,
            "eod_exit_time": "15:55"
        }
        
    return {
        "strategy": strat,
        "asset": asset,
        "timeframe": tf,
        "contracts": 2,
        "prop_preset": prop,
        "eval_risk": 1.0,
        "funded_risk": 0.5,
        "params": params,
        "explanation": f"Configured {strat} on {asset} ({tf}) with target '{params.get('target_mode', '2R')}', NY Open filter: '{params.get('ny_open_filter', 'none')}', Trend filter: '{params.get('trend_filter', 'none')}'."
    }

from engine.data_manager import normalize_time_str

def sanitize_and_validate_config(config_dict):
    """
    STRICT IMMUTABILITY GUARDRAIL:
    Ensures that only valid parameters from the schema are accepted,
    and strips out any attempts to modify core engine execution code.
    """
    strat = config_dict.get("strategy", "Value Area Reclaim")
    if strat not in STRATEGY_SCHEMA:
        strat = "Value Area Reclaim"
        
    schema = STRATEGY_SCHEMA[strat]
    raw_params = config_dict.get("params", {})
    clean_params = {}
    
    for key, spec in schema.items():
        val = raw_params.get(key, spec["default"])
        
        # Type & constraint checking
        if spec["type"] == "str":
            val = str(val)
            # If this is a time string field, normalize it first (handles '9:40', '9:40 am', etc.)
            if "pattern" in spec and spec["pattern"] == r"^\d{2}:\d{2}$":
                val = normalize_time_str(val, default=spec["default"])
            if "options" in spec and val not in spec["options"]:
                val = spec["default"]
            if "pattern" in spec and not re.match(spec["pattern"], val):
                val = spec["default"]
        elif spec["type"] == "float":
            try:
                val = float(val)
                val = max(spec.get("min", val), min(spec.get("max", val), val))
            except (ValueError, TypeError):
                val = spec["default"]
        elif spec["type"] == "bool":
            val = bool(val)
            
        clean_params[key] = val
        
    asset = config_dict.get("asset", "MNQ")
    if asset not in ALLOWED_ASSETS:
        asset = "MNQ"
        
    tf = config_dict.get("timeframe", "15m")
    if tf not in ALLOWED_TIMEFRAMES:
        tf = "15m"
        
    prop = config_dict.get("prop_preset", "Topstep $50k")
    if prop not in ALLOWED_PROPS:
        prop = "Topstep $50k"
        
    contracts = int(config_dict.get("contracts", 2))
    contracts = max(1, min(20, contracts))
    
    eval_risk = float(config_dict.get("eval_risk", 1.0))
    eval_risk = max(0.25, min(3.0, eval_risk))
    
    funded_risk = float(config_dict.get("funded_risk", 0.5))
    funded_risk = max(0.1, min(2.0, funded_risk))
    
    return {
        "strategy": strat,
        "asset": asset,
        "timeframe": tf,
        "contracts": contracts,
        "prop_preset": prop,
        "eval_risk": eval_risk,
        "funded_risk": funded_risk,
        "params": clean_params,
        "explanation": config_dict.get("explanation", "Setup successfully compiled.")
    }

def compile_user_prompt_to_setup(user_text, api_key=None, provider="openai"):
    """
    Parses a user natural language prompt into a strictly validated trading setup.
    If an API key is available, calls LLM via OpenAI API (or Gemini).
    Otherwise falls back to the deterministic pattern parser.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        # Use built-in heuristic compiler
        raw = parse_with_rule_fallback(user_text)
        return sanitize_and_validate_config(raw)
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return sanitize_and_validate_config(parsed)
    except Exception as e:
        # Graceful fallback to deterministic parser on API error
        raw = parse_with_rule_fallback(user_text)
        validated = sanitize_and_validate_config(raw)
        validated["explanation"] += f" (Note: compiled using built-in parser: {e})"
        return validated
