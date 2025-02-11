# get the data
git clone https://github.com/allenai/WildBench.git
python scripts/paper/preprocess.py

wget -P data/ https://storage.googleapis.com/arena_external_data/public/clean_battle_20240629_public.json
mkdir results

# compute formality and sentiment
python scripts/paper/compute_classifiers.py

python scripts/paper/shared_biases_human.py --cores 1 # Table 1a
python scripts/paper/shared_biases_llm.py --cores 1 # Table 1b

python scripts/paper/sample_efficient_cat.py --category1 is_chinese --cores 1 # Fig 2a
python scripts/paper/sample_efficient_cat.py --category1 is_code --cores 1 # Fig 2b
python scripts/paper/sample_efficient_cat.py --category1 is_hard --cores 1 # Fig 2c
python scripts/paper/sample_efficient.py --wildbench --cores 1 # Fig 3a
python scripts/paper/sample_efficient.py --cores 1 # Fig 3b
python scripts/paper/sample_efficient_cat.py --category1 is_code --category2 is_chinese --cores 1 # Fig 3c


# Run the last experiment: multivariate leaderboard
python scripts/paper/leaderboard_poly.py --cores 1 # Table 2
python scripts/paper/leaderboard_univariate.py # Table 3

# Run appendix experiment: Alternative rating systems
python scripts/paper/alternatives.py # Fig 4