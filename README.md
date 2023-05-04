Download Link: https://assignmentchef.com/product/solved-cs7638-assignment-1-regression
<br>



<h1>1        General Instructions</h1>

In this assignment, you will solve several tasks applying regression. For the first one, you need to try different regression models to fill the missing values in each feature with a regression model for each of them. For the second task, you will explore a dataset of email spam and train logistic regression to find junk email, and finally, conclude features that influence the decision the most.

You are required to submit your solutions via Moodle as a single zip file. The zip archive should contain a single ipynb, and a single PDF for the theoretical parts 2.2 and 3.2. Please, put your name and email at Innopolis.university as the first line in the notebook.

Source code should be clean, easy to read, and well documented. Bonus points may be awarded for elegant solutions. However, these bonus points will only be able to cancel the effect of penalties.

Do not just copy and paste solutions from the Internet. You are allowed to collaborate on general ideas with other students as well as consult books and Internet resources. However, be sure to credit the sources you use and type all the code, documentation by yourself.

<h1>2        Linear/Polynomial Regression</h1>

<h2>2.1       Practical Task 1 [25%]</h2>

In this task you are going to make data imputation. You are to fill all the missing values in the dataset. That is an important step towards solving any ML problem because missing data decrease accuracy of your model. You have a task1 dataset.csv file where a large percentage of data is lost. Also there is a ground truth dataset task1 dataset full.csv. There are 4 columns: datetime and 3 numerical features. The 3 numerical features are independent from each other. So you should not use one feature to estimate values of another. Fit models with datetime as X and feature as Y. Your task:

<ul>

 <li>Preprocess and visualize the dataset: [20%]

  <ul>

   <li>Encode datetime column with integer values from 0 to len(dataset). It will be easier to do visualization and model training.</li>

   <li>Plot all features of the dataset (on separate plots). Use matplotlib.pyplot for that.</li>

  </ul></li>

 <li>Use different regression models with different degrees from this interval [1<em>,</em>10] to predict missing values and fill the gaps (provide imputation). Don’t use imputation libraries. [50%]</li>

 <li>Plot change of MSE for each degree for all features. (MSE between imputed dataset and ground truth one). [20%]</li>

 <li>After experiments with regression models, report best regression degree for each feature. Explain this result: write your ideas on why these degrees best describe particular feature. [10%]</li>

</ul>

<h2>2.2       Theoretical Question On Ridge Regression [20%]</h2>

In the case of 1D data, the ridge regression estimator produces:

<em>n</em>

(<em>θ,</em><sup>ˆ </sup><em>θ</em><sup>ˆ</sup><sub>0</sub>) = <em>argmin</em><sup>X</sup>(<em>y<sub>t </sub></em>− <em>θx<sub>t </sub></em>− <em>θ</em><sub>0</sub>)<sup>2 </sup>+ <em>λθ</em><sup>2                                                                       </sup>(1)

<em>θ,θ</em><sub>0</sub>

<em>t</em>=1

for some <em>λ &gt; </em>0, and then makes predictions of the form ˆ<em>y </em>= <em>θx</em><sup>ˆ </sup>+<em>θ</em><sup>ˆ</sup><sub>0</sub>. In this question, we consider a 1D data set with 8 points (, marked with ‘X – a cross’ in the following figure:

The four dashed lines, labeled (A), (B), (C), and (D), each correspond to a linear prediction rule: Given a new <em>x</em>, each model predicts <em>y </em>to be the corresponding point on the line. For each of (A), (B), (C), and (D), indicate which of the following statements is the most appropriate:

<ul>

 <li><strong>High </strong><em>λ</em><strong>. </strong>The prediction rule could be produced by ridge regression with a high <em>λ </em>value;</li>

 <li><strong><u>Low </u></strong><em><u>λ</u></em><strong><u>. </u></strong>The prediction rule could be produced by ridge regression with a low <em>λ </em>value;</li>

 <li><u> </u>The prediction rule could not be produced by ridge regression.</li>

</ul>

Choose only one of these options for each of (A), (B), (C), and (D), and include a brief explanation (ideally only 1-2 sentences with minimal use of equations) for your choice.

<h1>3           Logistic Regression</h1>

<h2>3.1       Practical Task 2 [35%]</h2>

In this task you’re going to build a model for loan applicant approval. The goal is to classify loan applicants into one of two categories, good or bad. The dataset contains 1000 records of bank information of applicants represented with 20 attributes (7 numerical, 13 categorical). A full description of the dataset is attached.

Your task:

<ol>

 <li>Preprocess and visualize the dataset: Transform all categorical values into numerical values. You are free to apply any ways of handling categorical data and missing values.

  <ul>

   <li>Scale features if necessary. Explain why did you decide to scale or not.</li>

   <li>Visualize the dataset in two dimensions. Dimension reduction methods such as <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">PCA</a> can be used.</li>

   <li>Using pandas built-in function, plot the correlation matrix. Answer questions: Are there highly co-related features in the dataset? Is it a problem for regression task?</li>

  </ul></li>

 <li>Split the dataset into train(70%) and test set(30%).</li>

 <li>Apply logistic regression using linear and non linear function. Use different polynomial models with different degree (range from 1 to 10) and select the model which performs the best in terms of biasvariance. Highlight which model (degree) underfit and overfit the data <em>hint: plot the train and test error for each model</em>.</li>

 <li>Now that you have the best degree for your model, you will make it even better by fine-tuning its hyperparameters. Use GridSearchCV to find the best hyperparameters. Try different variations with penalty [’<em>l</em>1’, ’<em>l</em>2’], type of solver: [’liblinear’, ’lbfgs’], and regularization strength: <em>logspace</em>(4<em>,</em>4<em>,</em>20).</li>

 <li>Using your best model, compare the accuracy of predictions across male and female applicants e.i, split the test set into two groups (Male and female) compute the accuracy on each groups using the test set, plot and compare them. What conclusion can you draw and what could be the source of your observation?</li>

</ol>

<h2>3.2       On Regularization in Logistic Regression [20%]</h2>

In this problem we will refer to the binary classification task depicted in Figure 1a, which we attempt to solve with the logistic regression model

(for simplicity we do not use the intercept parameter <em>θ</em><sub>0</sub>). The training data can be separated with zero training error – see line <em>L</em><sub>1 </sub>in Figure 1b, for instance, which is the line obtained with no regularization. Consider a regularization approach where we try to maximize

for large <em>C</em>. Note that only <em>θ</em><sub>2 </sub>is penalized. Recall also that line <em>L</em><sub>1 </sub>in the figure corresponds to <em>C </em>= 0. We would like to know which of the four lines in Figure 1b could arise as a result of such regularization. For each potential boundary <em>L</em><sub>2</sub>, <em>L</em><sub>3</sub>, and <em>L</em><sub>4</sub>, determine whether it can result from regularizing <em>θ</em><sub>2</sub>. If not, explain briefly why not.

(b) The points can be separated by <em>L</em><sub>1 </sub>(solid line).

Possible other decision boundaries are shown by <em>L</em><sub>2</sub>,

(a) The 2D dataset used in Problem (3.2).                                       <em>L</em><sub>3</sub>, <em>L</em><sub>4</sub>.

Figure 1: Binary classification.

<h1>4        Notes</h1>

<ul>

 <li>Cheating is a serious academic offense and will be strictly treated for all parties involved. So delivering nothing is always better than delivering a copy.</li>

 <li>Late assignments will not be accepted and will receive <strong>ZERO </strong></li>

 <li>Code cleanness and style are assessed. So maybe you want to take a look at our references: <a href="https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html">Link 1</a> and <a href="https://www.python.org/dev/peps/pep-0008/">Link 2.</a></li>

 <li>Organize your notebook appropriately. Divide it into sections and cells with clear titles for each task and subtask.</li>

</ul>