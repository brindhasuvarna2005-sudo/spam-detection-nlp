### Dataset Overview

The combined dataset consists of both SMS and email messages used for spam detection.

- Total samples: 10,162
- SMS messages: 5,169
- Email messages: 4,993

The dataset contains both spam and ham messages, with a moderate class imbalance:
- Ham (0): 8,047
- Spam (1): 2,115

## cleaned dataset
Total samples: 10052

Source distribution:
source
sms      5158
email    4894
Name: count, dtype: int64

Label distribution:
label
0    7948
1    2104
Name: count, dtype: int64

## Experimental Setup

The dataset was divided based on source (SMS and Email) to evaluate both within-domain and cross-domain performance. A pre-trained DistilBERT model was fine-tuned for binary classification (spam vs ham).

Additionally, a baseline model using TF-IDF with Multinomial Naive Bayes was implemented for comparison.

Evaluation metrics used include accuracy, precision, recall, and F1-score.

## In-Domain Performance (SMS & Email)

The DistilBERT model demonstrates strong performance in in-domain settings for both SMS and email datasets. 

For SMS-only classification, the model achieves an accuracy of 99% with an F1-score of 0.97 for the spam class. The confusion matrix indicates very few misclassifications, with only 6 false negatives and 2 false positives. This shows that the model is highly effective in handling short and informal text messages.

In the email domain, the model achieves near-perfect performance with an accuracy of 100% and an F1-score of 0.99 for spam detection. This superior performance can be attributed to the structured and context-rich nature of email data, which allows the model to better capture semantic patterns.

The comparison between DistilBERT and the Naive Bayes baseline highlights the advantage of transformer-based models in spam detection tasks.

The comparison between DistilBERT and the Naive Bayes baseline highlights the advantage of transformer-based models in spam detection tasks.

## Baseline Comparison (DistilBERT vs Naive Bayes)
For SMS data, Naive Bayes achieves an accuracy of 96%, but its recall for spam detection is significantly lower (0.68), indicating that it fails to identify a substantial number of spam messages. In contrast, DistilBERT achieves a recall of 0.95, demonstrating its ability to capture contextual information more effectively.

Similarly, in the combined dataset, Naive Bayes achieves an accuracy of 95% with a spam recall of 0.82, whereas DistilBERT achieves 98% accuracy and a higher recall of 0.94. This suggests that traditional machine learning models struggle with diverse and mixed-domain data, while transformer-based models generalize better.

## Combined Dataset Performance
The combined dataset experiment shows that DistilBERT maintains strong performance even when trained on heterogeneous data. The model achieves an accuracy of 98% and an F1-score of 0.96 for spam classification.

However, compared to the email-only setting, a slight drop in performance is observed. This indicates that combining datasets introduces variability in language patterns, making the classification task more challenging. Despite this, the model remains robust and significantly outperforms the baseline.

## Several important observations can be drawn from the experimental results:

1. DistilBERT consistently outperforms Naive Bayes across all datasets, particularly in terms of recall for spam detection.
2. Email data yields higher performance compared to SMS data, likely due to its structured nature and richer context.
3. The combined dataset introduces additional complexity, resulting in a slight performance drop, but improves overall robustness.
4. Traditional models such as Naive Bayes struggle with detecting spam in diverse datasets, especially when contextual understanding is required.

## Interpretation

These findings highlight the importance of using deep learning approaches for real-world spam detection systems, where data variability and domain differences are significant challenges.
The results indicate that transformer-based models are highly effective for spam detection tasks due to their ability to capture contextual and semantic relationships within text. In contrast, classical models relying on bag-of-words representations are limited in handling variations in language, especially in informal or short messages.

Overall, the experimental results demonstrate that while high accuracy can be achieved in domain-specific settings, model performance is influenced by data characteristics, emphasizing the need for robust approaches that can handle diverse communication formats.

##  CROSS-DOMAIN RESULTS & DISCUSSION
# Cross-Domain Performance (SMS ↔ Email)

To evaluate the generalization capability of the models, cross-domain experiments were conducted by training on one domain and testing on another.

When trained on SMS and tested on email (SMS → Email), the DistilBERT model shows a significant drop in performance, achieving an accuracy of 53% and an F1-score of 0.31 for the spam class. The confusion matrix reveals a large number of misclassifications, with 951 false negatives and 1345 false positives. This indicates that the model struggles to correctly identify spam in the email domain when trained only on SMS data.

Similarly, in the Email → SMS setting, the model achieves an accuracy of 46%, with a spam recall of 0.74 but very low precision (0.15). This suggests that while the model is able to detect many spam messages, it incorrectly classifies a large number of legitimate messages as spam, leading to high false positives.


# Baseline Comparison in Cross-Domain

The Naive Bayes baseline also exhibits poor cross-domain performance. In the SMS → Email setting, it achieves an accuracy of 60%, but with a very low spam recall of 0.09, indicating that it fails to detect most spam messages.

In the Email → SMS setting, Naive Bayes achieves an accuracy of 59%, with improved recall (0.79) but very low precision (0.20), resulting in a large number of false positives.

Overall, both models show a substantial decline in performance in cross-domain scenarios, with DistilBERT slightly outperforming Naive Bayes in terms of balanced detection but still facing significant limitations.


# Key Insight: Domain Shift 


The sharp performance degradation observed in cross-domain experiments highlights the impact of domain shift. SMS and email messages differ significantly in terms of length, structure, vocabulary, and writing style.

Models trained on SMS data tend to learn short, informal patterns and fail to generalize to the more structured and context-rich nature of emails. Conversely, models trained on email data struggle with the brevity and noise present in SMS messages.

This demonstrates that spam detection models are highly sensitive to the characteristics of the training data and do not generalize well across different communication domains.


#  10. Error Analysis (Cross-Domain) 


Error analysis further reveals that misclassifications primarily occur in cases where messages contain ambiguous or mixed signals. In the SMS → Email setting, many legitimate emails are incorrectly classified as spam due to the presence of promotional language.

In the Email → SMS setting, the model tends to over-predict spam, leading to a high number of false positives. This is likely due to the model relying on strong spam-related keywords learned from longer email texts, which do not translate effectively to shorter SMS messages.

Additionally, shorter messages with limited context are more prone to misclassification, as the model lacks sufficient information to make accurate predictions.

# Final Interpretation (VERY IMPORTANT)


The cross-domain results clearly demonstrate that high in-domain accuracy does not guarantee generalization across different data distributions. While DistilBERT performs exceptionally well within individual domains, its performance drops significantly when applied to unseen domains.

This highlights a critical limitation of current spam detection approaches and emphasizes the need for domain-adaptive or multi-domain training strategies to improve robustness in real-world applications.







