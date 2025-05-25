Predicting Student Dropout Using Key Academic and Socioeconomic Indicators
 
Contributors:
- Daniel Herlev Moldavsky
- Jeppe Rønning Koch
- Laith Alkaseb
- Alfredo M. Fernandez
- Mads Benjamin Ribberholt


Streamlit Link: 
https://studentsuccesanalysis.streamlit.app/

Annotation:
In this project, we analyze data from 4,424 Portuguese students to find out which factors lead to dropout. By focusing on the most important variables (those with over 2% importance in a Random Forest model), we build machine learning models to predict dropout and explore how things like academic results, financial background, and age come into play. The goal is to help schools and universities catch at-risk students early and reduce dropout rates.

Problem Statement:
Can we predict a student's educational status (Dropout, Enrolled, or Graduate) based on the most important indicators like academic performance, financial background, and age – selected based on feature importance?

Purpose:
The goal is to build a model that can identify students at risk of dropping out – using only the most important features (those with over 2% importance in a Random Forest model) – so schools can take action in time.

 
Hypotheses (Aligned with Features)
Hypothesis 1: Academic Performance
H₀: The number of passed subjects and grades does not affect the likelihood of dropout.
H₁: Poor academic performance increases the likelihood of dropout.

Hypothesis 2: Economic Factors
H₀: Macroeconomic factors (GDP, unemployment, inflation) have no impact.
H₁: Macroeconomic conditions affect the likelihood of dropout.

Hypothesis 3: Other Key Factors
H₀: Age, previous grades, and parental background have no effect on dropout.
H₁: These factors significantly influence dropout.

Research Questions:

Descriptive Analysis

How do grades and study activity differ between dropouts and graduates?
Does financial background and age play a role in dropout?
Predictive Analysis

How accurate is the model when using only features with > 2% importance?
Which factors carry the most weight in the model’s decisions?
Which classification model performs best on this subset?
 
 
Final Project Conclusion:
In this project, we looked into why some students drop out of university by using data from Portuguese schools. The main idea was to build a model that can help spot students who might drop out early, based on how they perform in school, their background, and their economic situation. What We Found Grades Matter – a Lot Students who graduate usually start strong – they pass more subjects and get better grades already in the first semester. Those who drop out often struggle from the beginning. So yeah, academic performance early on really makes a difference. Age and Money Are Also Important Dropouts are more likely to be older when they start and often behind on tuition payments. Maybe they have jobs, families, or other stuff going on. It seems like both age and financial pressure make it harder to stay in school. The Economy Plays a Role Too We saw that students are more likely to drop out during times of low GDP and high unemployment. So, the bigger economic picture might also affect whether people stay in school. Background and Family Count Older age, lower grades from previous education, and lower parental education levels all showed clear differences between dropouts and graduates. So, where you come from and what kind of support you have can matter a lot. Model Results We tried a few machine learning models and got decent results: Our Decision Tree got about 73% accuracy and was good at spotting both graduates and dropouts. The KNN model did slightly better with 73% accuracy, especially good at finding graduates and dropouts. But the "Enrolled" group (students still studying) was hard to predict – probably because they’re somewhere in between. Wrapping Up We can say for sure that dropping out isn’t random. Students who leave early often have a tough time academically and maybe outside of school too. Using models like ours, schools could get better at finding students who need help before it’s too late. This project shows that BI and AI can actually be useful tools in education – not just for numbers and dashboards, but to support real people.
