# Gender Bias Classification with ALBERT

### **Table of Contents**
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
   - [Data Exploration and Class Distribution Analysis](#1-data-exploration-and-class-distribution-analysis)
   - [Passage Length Analysis](#2-passage-length-analysis)
3. [Model Selection and Iterative Optimization](#model-selection-and-iterative-optimization)
   - [Initial Model Selection: ALBERT](#1-initial-model-selection-albert)
   - [Hyperparameter Tuning](#2-hyperparameter-tuning)
   - [Model Optimization and Quantization](#3-model-optimization-and-quantization)
   - [SMOTE for Imbalance Handling](#4-smote-for-imbalance-handling)
4. [Model Evaluation](#model-evaluation)
5. [Training and Validation Journey](#training-and-validation-journey)
6. [Model Evaluation and Submission](#model-evaluation-and-submission)
7. [Plots and Visualizations](#plots-and-visualizations)
8. [Conclusion](#conclusion)
9. [Future Directions](#future-directions)
10. [Contact Information](#contact-information)

### **Introduction**

The aim of this competition was to advance the classification of gender bias in text using machine learning and deep learning models. Gender bias, manifesting in multiple forms, affects individuals across the gender spectrum, including males, females, and non-binary individuals. The objective was to build a model that not only accurately classifies gender bias but also maintains efficiency by minimizing the Number of Parameters (NOP). To meet the competition's unique **F1NOP** metric, we sought a balance between **accuracy** and **model complexity**.

This documentation captures the multiple iterations and techniques applied to reach an optimal solution using **ALBERT** while reducing computational complexity. Here, I will walk you through the various stages of the solution, including **data preprocessing, model tuning, parameter optimization, SMOTE for imbalance handling, quantization**, and how we ensured compliance with all competition requirements.

### **Data Preprocessing**

#### **1. Data Exploration and Class Distribution Analysis**

The initial step involved exploring the **train.csv** dataset to understand the class distribution and characteristics of the passages. We loaded the dataset using `pandas` and conducted an analysis to identify potential class imbalances. The dataset contained **10,000 passages**, and the classes represented gender bias against **Males (0)**, **Females (1)**, **Non-binary (2)**, and **Neutral (3)**.

**Class Distribution Visualization**

To visualize the class distribution, we plotted the frequency of each label:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load training dataset
train_df = pd.read_csv('train.csv')

# Plot class distribution
sns.countplot(x='y', data=train_df)
plt.xlabel('Bias Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in Training Data')
plt.savefig('plots/class_distribution.png')
```

This plot revealed that there was some imbalance between the classes, with **Neutral (3)** being overrepresented compared to other bias categories. This insight informed our decision to use **class weights** during model training to handle this imbalance.

#### **2. Passage Length Analysis**

We analyzed the length of passages to determine an appropriate **maximum sequence length** for tokenization. This helped in optimizing memory usage and avoiding unnecessary padding or truncation.

**Passage Length Distribution**

```python
# Analyze passage lengths
train_df['passage_length'] = train_df['passage'].apply(lambda x: len(x.split()))

# Plot distribution of passage lengths
plt.hist(train_df['passage_length'], bins=50, color='blue', alpha=0.7)
plt.xlabel('Passage Length (words)')
plt.ylabel('Frequency')
plt.title('Distribution of Passage Lengths')
plt.savefig('plots/passage_length_distribution.png')
```

Based on this analysis, we determined that a **maximum sequence length of 128 tokens** would be sufficient to capture most passages without excessive truncation.

### **Model Selection and Iterative Optimization**

#### **1. Initial Model Selection: ALBERT**

We began with the **ALBERT (A Lite BERT)** model, which was chosen for its reduced parameter space and computational efficiency compared to other large language models like BERT. The tokenizer used was **AlbertTokenizer**, initialized with the `albert-base-v2` configuration.

The dataset was split into **80% training** and **20% validation**, which allowed us to properly evaluate the model during training. During the initial training, we faced issues with **overfitting**, prompting us to implement **early stopping** to ensure the model would stop training once it began to diverge in performance on the validation set.

#### **2. Hyperparameter Tuning**

- **Learning Rate**: We experimented with different learning rates, and eventually set it to **3e-5**. This value provided stability during training and prevented the model from oscillating or getting stuck in local minima.
- **Batch Size**: After multiple experiments, we set the **batch size** to **32**. Increasing the batch size improved training stability and allowed better utilization of the available GPU memory.
- **Scheduler**: We used **linear scheduling with warm-up**, which helped maintain a smooth reduction in the learning rate over time.

**Training Results: Initial Iteration**

- Training Loss: Started around **0.4158** and improved with each epoch.
- Validation F1 Score: Peaked at **0.9603** after three epochs, which was impressive for our initial model.

#### **3. Model Optimization and Quantization**

##### **Class Weight Calculation**

- Due to the potential class imbalance in our dataset, **class weights** were computed using `compute_class_weight` from `sklearn`. This ensured that underrepresented classes were properly considered during training, leading to balanced performance across all gender bias categories.

##### **Quantization**

- To further reduce model complexity and achieve a lower **NOP**, we applied **dynamic quantization** using `torch.quantization.quantize_dynamic`. This effectively compressed the model by converting some operations to a more compact form without significant loss in accuracy.
- The quantized model maintained a **Validation F1 Score of 0.9355**, while reducing the number of parameters to **3.9 million**.

**Quantization Benefits**

- Reduced inference time, making the model more efficient for deployment.
- Lowered **NOP**, improving the **F1NOP** score per the competition requirements.

#### **4. SMOTE for Imbalance Handling**

To address the imbalance between the classes more effectively, we applied **Synthetic Minority Over-sampling Technique (SMOTE)** with **TF-IDF** representations to oversample the minority classes. This step ensured the model received a balanced representation of all classes during training, resulting in improved generalization.

- **TF-IDF Vectorization**: We transformed text data using `TfidfVectorizer` to generate numerical representations of the text.
- **SMOTE Application**: Using `imblearn`'s `SMOTE`, we balanced the classes before feeding the data into the training model.

This step helped increase the representation of underrepresented categories such as **Non-binary** and **Neutral**.

### **Model Evaluation**

We explored various models throughout our journey, including **BERT**, **TinyBERT**, **DistilBERT**, and **ALBERT**. Here is a summary of our findings for each model:

| Model                 | Batch Size | Learning Rate | Validation F1 Score | Number of Parameters (NOP) | Quantization Applied |
| --------------------- | ---------- | ------------- | ------------------- | -------------------------- | -------------------- |
| BERT                  | 16         | 5e-5          | 0.9392              | 110 million                | No                   |
| DistilBERT            | 32         | 3e-5          | 0.9538              | 66.9 million               | No                   |
| TinyBERT              | 16         | 5e-5          | 0.9434              | 14.4 million               | No                   |
| ALBERT (v1)           | 16         | 5e-5          | 0.9549              | 11.6 million               | No                   |
| ALBERT (v2)           | 32         | 3e-5          | 0.9603              | 11.6 million               | No                   |
| ALBERT (v3)           | 32         | 3e-5          | 0.9647              | 11.6 million               | No                   |
| ALBERT (v3) Quantized | 32         | 3e-5          | 0.9355              | 3.9 million                | Yes                  |

The table shows that the **quantized version of ALBERT** provided the best combination of **F1 Score** and **Number of Parameters**, making it the optimal model for submission.

### **Training and Validation Journey**

Below, we document the training journey and the changes in **training loss**, **validation loss**, and **validation F1 score** over each epoch:

- **Epoch 1**:

  - Training Loss: **0.2987**
  - Training F1 Score: **0.7617**
  - Validation Loss: **0.1113**
  - Validation F1 Score: **0.9093**

- **Epoch 2**:

  - Training Loss: **0.0752**
  - Training F1 Score: **0.9328**
  - Validation Loss: **0.0992**
  - Validation F1 Score: **0.9194**

- **Epoch 3**:

  - Training Loss: **0.0444**
  - Training F1 Score: **0.9584**
  - Validation Loss: **0.0649**
  - Validation F1 Score: **0.9355**

The **early stopping** mechanism prevented overfitting and ensured that the model was not trained beyond its optimal performance. This allowed us to achieve the highest possible F1 score while maintaining generalizability.

### **Classification Report and Confusion Matrix**

After training, the model was evaluated using a **classification report** and a **confusion matrix** to gain insights into its performance across different classes.

**Classification Report**:

The final classification report of the quantized model on the validation set showed:

- **Precision, Recall, and F1 Scores** for each class (Male, Female, Non-binary, Neutral).
- **Weighted Average F1 Score**: **0.9355**.

**Confusion Matrix**:

The confusion matrix was generated to illustrate the correct and incorrect predictions made by the model:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Male', 'Female', 'Non-binary', 'Neutral'], yticklabels=['Male', 'Female', 'Non-binary', 'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Gender Bias Classification')
plt.savefig('plots/confusion_matrix.png')
```

The confusion matrix revealed how well the model was distinguishing between different forms of gender bias, with specific attention to areas that required improvement.

### **Model Evaluation and Submission**

The **test set** was used for final evaluation and submission preparation. After training, the quantized model was used for inference to ensure minimal computational complexity during the test phase.

- **Number of Parameters (NOP)**: The NOP was calculated consistently using the **`thop`** package, as specified in the competition guidelines. The final **NOP** for our quantized model was **3.9 million**, which contributed positively to the **F1NOP** metric.

- **Submission File Insights**:
  - **NOP**: 3,909,120
  - **y-pred Distribution**: 
    - Class 0 (Male): **237**
    - Class 1 (Female): **1144**
    - Class 2 (Non-binary): **199**
    - Class 3 (Neutral): **1424**
  - **Total y-pred**: **3000**

### **Plots and Visualizations**

**Training and Validation Loss Plots**

To visualize the progress of training, we plotted the **training loss** and **validation loss** over the epochs. These plots helped us monitor whether the model was overfitting or underfitting:

```python
# Plotting metrics after training
epochs = list(range(1, len(training_losses) + 1))

# Plot training and validation loss
plt.plot(epochs, training_losses, label='Training Loss', marker='o')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig('plots/training_validation_loss.png')
```

**Validation F1 Score Plot**

We also plotted the **Validation F1 Score** over the epochs to track improvements in the model's classification performance:

```python
# Plot validation F1 score
plt.plot(epochs, validation_f1_scores, label='Validation F1 Score', color='green', marker='o')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score Over Epochs')
plt.legend()
plt.savefig('plots/validation_f1_score.png')
```

**Confusion Matrix Plot**

The confusion matrix was visualized to highlight the performance across different bias categories:

```python
# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Male', 'Female', 'Non-binary', 'Neutral'], yticklabels=['Male', 'Female', 'Non-binary', 'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Final Model')
plt.savefig('plots/final_confusion_matrix.png')
```

These visualizations were instrumental in helping us decide when to stop training and in understanding the impact of our hyperparameter adjustments and SMOTE.

### **Conclusion**

This journey to develop a robust gender bias classification model using **ALBERT** involved a meticulous process of **data exploration, model training, hyperparameter tuning, SMOTE application for class imbalance, and quantization**. The final quantized ALBERT model achieved a balance between **high accuracy** and **low computational complexity**, making it suitable for deployment in real-world scenarios where efficiency is critical.

The **Validation F1 Score** of **0.9355** and the **number of parameters (NOP) of 3.9 million** underscored the effectiveness of the model in achieving the competition’s requirements for accuracy and efficiency. Our model was able to effectively identify gender bias across multiple classes while keeping the model size manageable.

Our experience demonstrated that achieving a competitive **F1NOP** score required not only model performance optimization but also careful attention to reducing model size through techniques like **quantization** and handling data imbalances with **SMOTE**. This combination of methods allowed us to create a solution that met both the accuracy and efficiency requirements of the competition.

### **Future Directions**

*To further improve, future iterations could explore:*

- **Knowledge Distillation**: Training a smaller model based on the outputs of ALBERT, which can lead to a more lightweight model without significant losses in performance.
  
- **Data Augmentation**: Expanding the dataset with more examples, especially for underrepresented classes, to potentially improve generalizability and robustness.

- **Advanced Quantization Techniques**: Exploring post-training quantization or quantization-aware training to further reduce the computational burden and enhance deployment efficiency.

- **Ensemble Techniques**: Leveraging multiple lightweight models or combining different versions of ALBERT could potentially enhance prediction accuracy and robustness.

### **Contact Information**

For further questions or collaboration opportunities, feel free to contact me:

- **Email**: ahmad.konainah@gmail.com
- **LinkedIn**: [Ahmad's LinkedIn](https://linkedin.com/in/ahmad-konainah)
