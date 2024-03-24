api_key = "--YOUR-CHATGPT_API--"

BASE_DATA_FOLDER = "/home/atif/work/notebook-data/data/"

# 3 datasets: Toraman -> LSHS, Hateval -> Hateval, Vidgen -> VIDH
LSHS_DATAFILE = BASE_DATA_FOLDER + 'LSHS-LREC-2022/lrec_tweets_hatespeech.csv'
HEVAL_DATAFILE = BASE_DATA_FOLDER + 'Hateval-2019/hateval2019_en_test.csv'
VIDH_DATAFILE = BASE_DATA_FOLDER + 'Vidgen-Hate-v-dot2-2021/vidgen_dataset.csv'
VIDH_MANUAL_CF = BASE_DATA_FOLDER + 'Vidgen-Hate-v-dot2-2021/vidgen_cfs_dataset.csv'

en_swear_words_datafile = BASE_DATA_FOLDER + 'uk-attitudes-to-offensive-language-and-gestures-data/data/list-of-swearwords-and-offensive-gestures.csv'
en_profanity_datafile =  BASE_DATA_FOLDER + 'profanity/profanity_en.csv'

gpt_response_lshs_file = 'out/gpt-response-lshs.json'
gpt_filtered_rephrase_lshs_file = 'out/gpt-filtered-rephrase-tweet-lshs.json'
pj_cad_lshs_file = 'out/pj-cad-tweet-lshs.json'

gpt_response_vidh_file = 'out/gpt-response-vidh.json'
gpt_filtered_rephrase_vidh_file = 'out/gpt-filtered-rephrase-tweet-vidh.json'
pj_cad_vidh_file = 'out/pj-cad-tweet-vidh.json'
gpt_pj_mixed_cad_vidh_file = 'out/gpt-pj-mixed-cad-tweet-vidh.json'
