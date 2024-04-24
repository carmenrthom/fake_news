# Fake News Binary Classifier using BERT

## How to run  
First, enter the directory of the project
```{bat}
cd fake_news
```

Install all dependencies 
```{bat}
pip install -r requirements.txt
```

Preprocess the data
```{bat}
python preprocessing.py
```

Run the model  
(Parameters for hypertuning are available in the main function)  
(More BERT models for MODEL parameter available [here](https://huggingface.co/models))

```{bat}
python model.py
```

Evaluation script TO BE IMPLEMENTED

