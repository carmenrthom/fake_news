# Fake News Binary Classifier using BERT

## How to run  

First clone the repository
```{bat}
git clone https://github.com/carmenrthom/fake_news.git
```

First, enter the directory of the project
```{bat}
cd fake_news
```

Install all dependencies 
```{bat}
pip install -r requirements.txt
```

Run the fine-tuning procedure for the model  
(Parameters for hypertuning are available in the main function)  
(More BERT models for MODEL parameter available [here](https://huggingface.co/models))
```{bat}
python model.py
```

Evaluation script can be run
```{bat}
python eval.py
```

