# SREPS

Implementation of "Social Recommendation with an Essential Preference Space".

If you find this method helpful for your research, please cite this paper:

```latex
@inproceedings{LiuZWHG18,
  author    = {Chun{-}Yi Liu and
               Chuan Zhou and
               Jia Wu and
               Yue Hu and
               Li Guo},
  title     = {Social Recommendation with an Essential Preference Space},
  booktitle = {Proceedings of the Thirty-Second {AAAI} Conference on Artificial Intelligence (AAAI-18)},
  pages     = {346--353},
  year      = {2018}
}
```

------

### Requirement

- python >= 3.4
- numpy
- tqdm

------

### Dataset

The dataset used in this paper can be obtained from the original papers.

| Dataset   | Paper                                                        |
| --------- | ------------------------------------------------------------ |
| FilmTrust | G. Guo, J. Zhang, and N. Yorke-Smith, “A novel bayesian similarity measure for recommender systems,” *in Proc of (IJCAI)*, 2013, pp. 2619–2625. |
| Flixster  | M. Jamali and M. Ester, “A matrix factorization technique with trust propagation for recommendation in social networks,” *in Proc. of RecSys*, 2010, pp. 135–142. |
| Epinions  | J. Tang, H. Gao, and H. Liu, “mtrust: discerning multi-faceted trust in a connected world,” *in Proc. of WSDM*, 2012, pp. 93–102. |
| Ciao      | J. Tang, H. Gao, H. Liu, and A. D. Sarma, “etrust: understanding trust evolution in an online world,” *in Porc. of KDD*, 2012, pp. 253–261. |

------

### How to use

#### Step 1. Format The Dataset

The four datasets are different in the input files on the social links. We first format them into a same format by:

```bash
python main.py --mode=prepro --dataset=filmtrust --data_input=xxxx --data_output=yyy
```

* The `dataset` is the name of dataset, and the `filmtrust`, `flixster`, `epinions` and `ciao` are supported. 

* The `data_input` is the path (a folder) of the input file. The preprocessing will read the file with the **predefined** name in the path. For `filmtrust`, `epinions` and `ciao`, the rating file (i.e. `ratings.txt`) and the trust link file (`trust.txt`) must exist. And for `flixster`, the rating file (i.e. `ratings.txt`) and link file (`links.txt`) must exist. The predefined name can be modified in the `preprocess.py`.
* The `data_out` is the path (a folder) to save the formated files. The rating file is saved as `${data_out}/ratings.total`, and the social link file is saved as `${data_out}/links.total`.

#### Step 2: Train The Model

In the **first run**, we can train our model by:

```bash
python main.py --mode=run --rating_file=xxxx --link_file=yyy --train_file=mmm --dev_file=nnn
```

The input ratings will first be divided into a training dataset and a evaluation dataset. And they are saved in the `${train_file}` and `${dev_file}` respectively. 

After the data division, the training will start. In the training process, we can observe the performance on the evaluation set.

For the **non-first** run, we can simply start with 

```bash
python main.py --mode=run --train_file=mmm --dev_file=nnn
```

 The `train_file` and `dev_file` are loaded, and the model will train with the `${train_file}`. The performance on the evaluation set can also be observed.

In the training process, the model will automatically save in the `save_path` every `save_step`. Note only **one** model are saved, and the users should monitor the performance and manually conduct the early stopping.

**Note**: other hyper-parameters of the model and the training process can be found in the `main.py`. 

#### Step 3: Evaluation.

Given the evaluation dataset, we can conduct the evaluation with:

```bash
python main.py --mode=eval --dev_file=xxx --save_path=yyy
```

The model will loaded from the `${save_path}`, and the performance of the evaluation set `${dev_file}` will be reported.

------

### Disclaimer

If you find any bugs, please report them to me.