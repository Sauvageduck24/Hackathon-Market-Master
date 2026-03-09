##################
# DEFAULT VALUES #
##################

DATA        := "data"

# Default time range

# dataset00
# START_TS    := "1746057600000"   # 2025‑05‑01
# END_TS      := "1746144000000"   # 2025‑05‑02

# dataset01
# START_TS    := "1738368000000"   # 2025‑02‑01
# END_TS      := "1743465600000"   # 2025‑04‑01

# dataset02
# START_TS    := "1612137600000"   # 2021‑02‑01
# END_TS      := "1617235200000"   # 2021‑04‑01

# dataset03
START_TS    := "1740614400000"   # 2023‑02‑01
END_TS      := "1745712000000"   # 2023‑04‑01

TIMEFRAME   := "1m"

# Default trading fees
FEE := "3"  # In basis points (3 = 0.03%)

# Default assets and balances
# TOKEN_1 := "ETH"
# TOKEN_2 := "BTC" 
# FIAT    := "USDT"

TOKEN_1 := "ETH"
TOKEN_2 := "BTC" 
FIAT    := "USDT"

TOKEN_1_BALANCE := "100"
TOKEN_2_BALANCE := "10"
FIAT_BALANCE    := "500000"

# Default team
TEAM        := "alpha"


###########
# RECIPES #
###########

# print options
default:
    @just --list --unsorted

alias i := install
alias b := build
alias d := download
alias t := tar
alias c := clean

# install Python requirements
install:
    pip install --upgrade pip && \
    pip install -r requirements.txt

# build the Docker environment
build:
    docker build -t junz .

# download market data for several pairs
download token1=TOKEN_1 token2=TOKEN_2 fiat=FIAT:
    #!/usr/bin/env bash
    # Convert to lowercase for filenames
    TOKEN1_LC=$(echo "{{token1}}" | tr '[:upper:]' '[:lower:]')
    TOKEN2_LC=$(echo "{{token2}}" | tr '[:upper:]' '[:lower:]')
    FIAT_LC=$(echo "{{fiat}}" | tr '[:upper:]' '[:lower:]')
    
    echo "Downloading data for {{token1}}/{{fiat}}, {{token2}}/{{fiat}}, and {{token1}}/{{token2}}..."
    
    # Create the data directory if it doesn't exist
    mkdir -p {{DATA}}
    
    # Download token1/fiat data
    python scripts/download.py "{{token1}}/{{fiat}}" --start {{START_TS}} --end {{END_TS}} \
        --output {{DATA}}/${TOKEN1_LC}${FIAT_LC}_{{TIMEFRAME}}.csv
    
    # Download token2/fiat data
    python scripts/download.py "{{token2}}/{{fiat}}" --start {{START_TS}} --end {{END_TS}} \
        --output {{DATA}}/${TOKEN2_LC}${FIAT_LC}_{{TIMEFRAME}}.csv
    
    # Download token1/token2 data
    python scripts/download.py "{{token1}}/{{token2}}" --start {{START_TS}} --end {{END_TS}} \
        --output {{DATA}}/${TOKEN1_LC}${TOKEN2_LC}_{{TIMEFRAME}}.csv

    echo "Download complete for {{token1}}, {{token2}}, and {{fiat}}."

    echo "Merging data files..."
    python scripts/merge.py \
        {{DATA}}/${TOKEN1_LC}${FIAT_LC}_{{TIMEFRAME}}.csv \
        {{DATA}}/${TOKEN2_LC}${FIAT_LC}_{{TIMEFRAME}}.csv \
        {{DATA}}/${TOKEN1_LC}${TOKEN2_LC}_{{TIMEFRAME}}.csv \
        --output {{DATA}}/test.csv \
        --token1 {{token1}} \
        --token2 {{token2}} \
        --fiat {{fiat}}
    echo "Data files merged into {{DATA}}/test.csv"

# archive the trading strategy
tar team=TEAM:
    @echo "Creating strategy archive for team {{team}}..."
    tar -czf {{team}}_submission.tgz strategy/

# generate transactions with a trading strategy
trade team=TEAM token1=TOKEN_1 token2=TOKEN_2 fiat=FIAT token1_balance=TOKEN_1_BALANCE token2_balance=TOKEN_2_BALANCE fiat_balance=FIAT_BALANCE fee=FEE:
    #!/usr/bin/env bash
    # Calculate strategy file name
    STRATEGY_FILE="{{team}}_submission.tgz"

    # Convert to lowercase for filenames
    TOKEN1_LC=$(echo "{{token1}}" | tr '[:upper:]' '[:lower:]')
    TOKEN2_LC=$(echo "{{token2}}" | tr '[:upper:]' '[:lower:]')
    FIAT_LC=$(echo "{{fiat}}" | tr '[:upper:]' '[:lower:]')

    echo "Trading strategy with {{token1}}/{{fiat}}, {{token2}}/{{fiat}}, and {{token1}}/{{token2}} for team {{team}}..."
    echo "Initial balances: {{token1}}={{token1_balance}}, {{token2}}={{token2_balance}}, {{fiat}}={{fiat_balance}}"
    FEE_DECIMAL=$(echo "scale=4; {{fee}}/10000" | bc)
    FEE_PERCENT=$(echo "scale=2; {{fee}}/100" | bc)
    echo "Trading fee: {{fee}} basis points ($FEE_DECIMAL or ${FEE_PERCENT}%)"

    python -m exchange.trade ${STRATEGY_FILE} \
        --data {{DATA}}/test.csv \
        --output {{DATA}}/submission.csv \
        --token1_balance {{token1_balance}} \
        --token2_balance {{token2_balance}} \
        --fiat_balance {{fiat_balance}} \
        --fee {{fee}}

# remove downloaded data and generated archives
clean:
    rm -f data/*.parquet && \
    rm -f *_submission.tgz