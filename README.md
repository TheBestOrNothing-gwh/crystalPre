# CrysXPP: An Explainable Property Predictor for Crystalline Materials

This is software package for Crsytal Explainable Property Predictor(CrysXPP) that takes as input
any arbitary crystal structure in .cif file format and predict different state and elastic properties
of the material.

It has two modules :

- Crystal Auto Encoder (CrysAE) : An auto-encoder based architecture which is trained with a large amount of unlabeled crystal data which leads to the deep encoding module capturing all the important structural and chemical information of the constituent atoms (nodes) of the crystal graph. 

    ![CrysAE diagram](images/CrysAE.png)
    <div align='center'><strong>Figure 1. CrysAE Architecure.</strong></div>
    
- Crystal eXplainable Property Predictor (CrysXPP) : An Explainable Property Predictor, to which the knowledge acquired by the encoder is transferred and which is further trained with a small amount of property-tagged data.

    ![CrysXPP diagram](images/CrysXPP.png)
    <div align='center'><strong>Figure 2. CrysXPP Architecure.</strong></div>

The following paper describes the details of the CrysXPP framework:

[CrysXPP: An Explainable Property Predictor for Crystalline Materials](https://arxiv.org/pdf/2104.10869.pdf)

## Table of Contents

- [CrysXPP: An Explainable Property Predictor for Crystalline Materials](#crysxpp-an-explainable-property-predictor-for-crystalline-materials)
  - [Table of Contents](#table-of-contents)
  - [How to cite](#how-to-cite)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Define a customized dataset](#define-a-customized-dataset)
    - [Train a CrysAE model](#train-a-crysae-model)
    - [Train a CrysXPP model](#train-a-crysxpp-model)
    - [Predict material properties with a pre-trained CrysXPP model](#predict-material-properties-with-a-pre-trained-crysxpp-model)
  - [Data](#data)
  - [Authors](#authors)
  - [License](#license)


## How to cite

If youare using CrysXPP, please cite our work as follow :

```
@article{das2022crysxpp,
  title={CrysXPP: An explainable property predictor for crystalline materials},
  author={Das, Kishalay and Samanta, Bidisha and Goyal, Pawan and Lee, Seung-Cheol and Bhattacharjee, Satadeep and Ganguly, Niloy},
  journal={npj Computational Materials},
  volume={8},
  number={1},
  pages={1--11},
  year={2022},
  publisher={Nature Publishing Group}
}
```


##  Requirements

The package requirements are listed in requirements.txt file. Run the following command to install dependencies in your virtual environment:

pip install -r requirements.txt

## Usage

### Define a customized dataset 
(Customiation is adopted From CGCNN Paper)

To input crystal structures to CrysAE and CrysXPP, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. 
2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There is a examples of customized dataset in the repository: `../data/`, with 37K cif files, where in id_prop file we have formation energy values.

### Train a CrysAE model

We have already trained the autoencoder with 37K data and a pretrained model (model_pretrain.pth) will be provided into the '../model' directory.

Yet, if You want to train the autoenoder module from scratch by some other dataset, use the following procedure :
- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.
- Run the following command

```bash
python -W ignore pretrain.py --data-path <Data_Path> --is-global-loss <1/0> --is-local-loss <1/0>
```
Once the training is done the saved model will be saved at "../model/model_pretrain.pth".

### Train a CrysXPP model
Before training a new CrysXPP model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

You can train the property predictor module with Pretrained Model + Feature Selector by the following command :

```bash
python -W ignore main.py --train-ratio 0.2 --val-ratio 0.2 --test-ratio 0.6 --data-path <Data_Path> --pretrained-model "../model/model_pretrain.pth" --epoch=200
```
As "pretrained-model" you can either use the existing pretarined CrysAE model "model/model_pretrain.pth" or you can pretrain your own  [CrysAE model](#train-a-crysae-model) and use the saved model.

If you want to train the CrysXPP property Predictor without Pretrained Model + Feature Selector (Exact CGCNN), use the following command:

```bash
 python -W ignore main.py --train-ratio 0.2 --val-ratio 0.2 --test-ratio 0.6 --data-path <Data_Path> --epochs=200 --feature-selector False
```
Here you can set set the following hyperparameters :

- lrate : Learning Rate (Default : 0.003).
- atom-feat : Atom Feature Dimension (Default : 64).
- nconv : Number of Convolution Layers (Default : 3).
- epoch : Number of Training Epochs (Default : 200)
- batch-size : Batch size of data (Default : 512).


After training, you will get following files :

- ``../model/model_best.pth.tar`` : Saved model for best property predictor.
-  ``../results/Prediction/<DATE>/<DATETIME>/out.txt`` : All the traing results for all epochs and all the hyperparameters are saved here.

### Predict material properties with a pre-trained CrysXPP model
Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.
- Obtain a pre-trained CrysXPP model saved in ``../model/model_pp.pth``.

Then, in directory src, you can predict the properties of the crystals in root_dir:

python predict.py --property-model <Pretrain_CrysXPP_path>

After predicting, you will get one file in src directory:

test_results.csv: stores the ID, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in id_prop.csv, which is not important

## Data

We have used the dataset provided by [CGCNN](https://github.com/txie-93/cgcnn). Please use the dataset to reproduce the results. CIF files are given in the "data/" directory and in id_prop file we have formation energy values.

## Authors

This software was primarily written by [Kishalay Das](https://kdmsit.github.io/) & [Bidisha Samanta](https://sites.google.com/view/bidisha-samanta/) 
and was advised by [Prof. Niloy Ganguly](http://www.facweb.iitkgp.ac.in/~niloy/), , [Prof. Pawan Goyal](https://cse.iitkgp.ac.in/~pawang/), Dr. Satadeep Bhattacharjee and Dr. Seung-Cheol Lee. 

## License

CrysXPP is released under the MIT License.
