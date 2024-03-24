# Inclusive-Counterfactual-Generation-Leveraging-LLMs-in-Identifying-Online-Hate

Repository for the paper "Inclusive Counterfactual Generation: Leveraging LLMs in Identifying Online Hate" accepted for publication at the ICWE 2024, 24th International Conference on Web Engineering, Tampere, Finland


## Files and Explanation

1. chatgpt-synthatic-\<dataset>.ipynb --> creates synthatic text using chatgpt api
2. gpt-json-proproces-\<dataset>.ipynb --> preprocess and extract synthatic tweets produced by chatgpt-synthatic-\<dataset>.ipynb
3. experiment-\<dataset>-crossfold.ipynb --> runs traditional ML pipelines (exp 1)
4. exp-\<dataset>_bert-FT.ipynb --> runs bert experimental setup (exp 2)
5. exp-\<dataset-1>_out-of_domain-\<dataset-2>_bert-FT.ipynb --> out of domain exp (a notebook each for exp2 & exp 3)
6. sota-\<name>.ipynb --> sota works
7. result-analysis.ipynb --> displays experimental results in full
