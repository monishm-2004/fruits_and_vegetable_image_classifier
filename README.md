# Image Classification Project with DVC and AWS S3

A complete machine learning project for classifying fruits and vegetables using CNN with DVC (Data Version Control) and AWS S3 for data versioning and remote storage.

## 🚀 Features

- **CNN Model**: Deep learning model for image classification of 36 fruit/vegetable categories
- **DVC Integration**: Version control for datasets and models
- **AWS S3 Storage**: Remote storage for large data files
- **Reproducible Pipeline**: Automated training pipeline with DVC
- **Streamlit Web App**: Interactive web interface for predictions
- **Experiment Tracking**: Track metrics and model performance

## 📊 Dataset

The dataset contains images of 36 different fruits and vegetables:
- **Training**: Used to train the model
- **Validation**: Used during training for hyperparameter tuning
- **Test**: Used for final model evaluation

Categories include: apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip, watermelon.

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Image_classification
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up AWS Credentials

Configure AWS credentials (see [AWS_S3_SETUP.md](AWS_S3_SETUP.md)):

```bash
aws configure
```

Or set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## 🗄️ DVC Setup

### Quick Setup (Automated)

```bash
chmod +x setup_dvc.sh
./setup_dvc.sh
```

### Manual Setup

```bash
# Initialize DVC
dvc init

# Add S3 remote storage
dvc remote add -d myremote s3://your-bucket-name/image-classification
dvc remote modify myremote region us-east-1

# Track data with DVC
dvc add Fruits_Vegetables
dvc add Image_classify

# Commit DVC files
git add .dvc .gitignore *.dvc
git commit -m "Initialize DVC with S3"

# Push data to S3
dvc push
```

For detailed instructions, see [AWS_S3_SETUP.md](AWS_S3_SETUP.md).

## 🔄 DVC Pipeline

The project includes a reproducible ML pipeline defined in `dvc.yaml`:

### Pipeline Stages

1. **Data Preparation**: Prepares and validates datasets
2. **Training**: Trains the CNN model
3. **Evaluation**: Evaluates model on test set

### Run the Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# View pipeline DAG
dvc dag
```

### Modify Parameters

Edit `params.yaml` to change hyperparameters:

```yaml
train:
  epochs: 25
  batch_size: 32
  learning_rate: 0.001
```

Then re-run the pipeline:

```bash
dvc repro
```

## 📈 Experiment Tracking

### View Metrics

```bash
# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

### View Plots

```bash
# Show plots
dvc plots show

# Compare training curves
dvc plots diff
```

## 🎯 Model Training

### Using the Pipeline (Recommended)

```bash
dvc repro
```

### Manual Training

```bash
python scripts/train.py
```

### Using Jupyter Notebook

```bash
jupyter notebook Image_Class_Model.ipynb
```

## 🌐 Web Application

### Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Usage

1. Enter the image filename in the text input
2. The app will display the image and prediction
3. Shows the predicted category and confidence score

## 📁 Project Structure

```
Image_classification/
├── Fruits_Vegetables/          # Dataset (tracked by DVC)
│   ├── train/                  # Training images
│   ├── validation/             # Validation images
│   └── test/                   # Test images
├── scripts/                    # Training scripts
│   ├── prepare_data.py         # Data preparation
│   ├── train.py                # Model training
│   └── evaluate.py             # Model evaluation
├── models/                     # Trained models (tracked by DVC)
├── metrics/                    # Evaluation metrics
├── plots/                      # Training plots
├── results/                    # Prediction results
├── .dvc/                       # DVC configuration
│   ├── config                  # DVC remote config
│   └── config.local            # Local credentials (not in Git)
├── app.py                      # Streamlit web app
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Hyperparameters
├── requirements.txt            # Python dependencies
├── setup_dvc.sh               # DVC setup script
├── .gitignore                 # Git ignore rules
├── AWS_S3_SETUP.md            # S3 setup guide
└── README.md                  # This file
```

## 🔧 Configuration Files

### params.yaml

Contains all hyperparameters for reproducibility:
- Image dimensions
- Training parameters (epochs, batch size, learning rate)
- Model architecture parameters

### dvc.yaml

Defines the ML pipeline:
- Dependencies for each stage
- Parameters used
- Outputs and metrics
- Plots to generate

### .dvc/config

DVC configuration including S3 remote:
- Remote storage URL
- AWS region
- Authentication settings (encrypted)

## 🤝 Collaboration Workflow

### Initial Setup (First Team Member)

```bash
# Setup project
./setup_dvc.sh

# Commit changes
git add .
git commit -m "Setup DVC with S3"
git push

# Push data to S3
dvc push
```

### Joining the Project (Other Team Members)

```bash
# Clone repository
git clone <repo-url>
cd Image_classification

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Pull data from S3
dvc pull
```

### Making Changes

```bash
# Modify params.yaml or code
vim params.yaml

# Run experiments
dvc repro

# Compare results
dvc metrics diff
dvc plots diff

# Push new data/models
dvc push

# Commit changes
git add .
git commit -m "Experiment with new hyperparameters"
git push
```

## 📊 Model Architecture

```
Input (180x180x3)
    ↓
Rescaling (1/255)
    ↓
Conv2D (16 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dropout (0.2)
    ↓
Dense (128 units)
    ↓
Dense (36 units - output)
```

## 📝 Common Commands

### DVC Commands

```bash
# Pull data from S3
dvc pull

# Push data to S3
dvc push

# Run pipeline
dvc repro

# Check status
dvc status

# Show metrics
dvc metrics show

# Show plots
dvc plots show

# List tracked files
dvc list . --dvc-only
```

### Git + DVC Workflow

```bash
# Track new data
dvc add new_data_folder

# Commit to Git
git add new_data_folder.dvc .gitignore
git commit -m "Add new data"

# Push data to S3
dvc push

# Push code to Git
git push
```

## 🐛 Troubleshooting

### DVC Pull Fails

```bash
# Check remote configuration
dvc remote list -v

# Verify AWS credentials
aws s3 ls s3://your-bucket-name

# Re-configure remote if needed
dvc remote modify myremote url s3://your-bucket-name/path
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Not Found

```bash
# Pull models from S3
dvc pull

# Or retrain
dvc repro train
```

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [AWS S3 Setup Guide](AWS_S3_SETUP.md)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset source
- TensorFlow team
- DVC community
- AWS for S3 storage
