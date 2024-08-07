## Project Title : Email Campaign Effectiveness Prediction

#### Steps Performed :

- Import required packages
- Data set loading
- EDA
- Data Pre-processing
- Multiple model building/training
- Generating pickel file
- Initilazion of flask app
- Import Models
- Render html forms for user input
- Pass inputs to model
- Get json data from model and render back to html form.
- Deployed Model to Aws Cloud
- Launched ubuntu instance with docker
- Imported all content to docker container
- Run flask app
- Map the ip of docker container with ec2-instance
- Expose the ec2-instance
- Serve the model by public ip

###   Problem Description :

Most of the small to medium business owners are making effective use of Gmail-based Email marketing Strategies for offline targeting of converting their prospective customers into leads so that they stay with them in business. The main objective is to create a machine learning model to characterize the mail and track the mail that is ignored; read; acknowledged by the reader. Data columns are self-explanatory.   

### Data Summary :

Our email campaign dataset have 68353 observations and 12 features. Clearly Email_Status is our target variable.

Our features:

- Email Id - It contains the email id's of the customers/individuals
- Email Type - There are two categories 1 and 2. We can think of them as marketing emails or important updates, notices like emails regarding the business.
- Subject Hotness Score - It is the email's subject's score on the basis of how good and effective the content is.
- Email Source - It represents the source of the email like sales and marketing or important admin mails related to the product.
- Email Campaign Type - The campaign type of the email.
- Total Past Communications - This column contains the total previous mails from the same source, the number of communications had.
- Customer Location - Contains demographical data of the customer, the location where the customer resides.
- Time Email sent Category - It has three categories 1,2 and 3; the time of the day when the email was sent, we can think of it as morning, evening and night time slots.
- Word Count - The number of words contained in the email.
- Total links - Number of links in the email.
- Total Images - Number of images in the email.
- Email Status - Our target variable which contains whether the mail was ignored, read, acknowledged by the reader.


### No of Models Trained :

- LogisticReg RUS
- LogisticReg SMOTE
- Decision Tree RUS
- Decision Tree SMOTE
- Random Forest RUS
- Random Forest SMOTE
- RandomF Tuned RUS
- RandomF Tuned SMOTE
- RandomF Tuned RUS FSel
- RandomF Tuned SMOTE FSel
- XGB RUS
- XGB SMOTE

### Final Model Selection :

We have selected best 3 Models which are:

- XGB SMOTE with accuracy 76 %
- RandomF Tuned SMOTE FSel with accuracy 75 %
- RandomF Tuned SMOTE with accuracy 75 % 

### Model Deployed Link:
[Click Here To Access Model](http://13.233.145.93/)

## Thank You...
