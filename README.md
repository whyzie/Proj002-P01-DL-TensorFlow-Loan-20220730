# Proj002-P01-DL-TensorFlow-Loan-20220730
Predict loan fully paid with tensorflow - keras

<img style="float:left" src="https://i.imgur.com/DWQOJgU.png" width="1500">

# Deep Learning - TensorFlow Keras API- Loan Data
<hr>
30 July 2022
<div>
<img style="float: left" src="https://i.postimg.cc/qvZpHzhk/002-Img-Objectives-Draft-2-20220819.png(https://postimg.cc/cvYpQ1Bj)" width="75">
<h2 id="Obj1">Objectives</h2>
<hr>

<h3 style="text-align:justify; font-family: Arial">Create a model that will predict whether a profile will pay back their loan(s) (accountable lenders). Data source<a href="https://www.lendingclub.com/"> LendingClub.com.</a>
<br><br>
Note: This project's dataset was created for pedagogical purposes and some data may differ being added or removed from the source.
  
<div class="alert alert-block alert-warning">
<img style="float: left" src="https://i.postimg.cc/kXz8cFqC/005-Img-Yellow-Notes-Draft-1-20220819.png" width="60">
<b style = "font-family: Arial; font-size: 16px">Remember:</b><p style = "font-family:Verdana; font-size:14px">We must be <b>objective</b> in analyzing the data in order to acquire valuable insights and understand it by collecting, fact checking or challenging the data and other sources. Go to where the data - Genchi Genbutsu attitudes. Data must be <b>Clear, Objective, Valuable, Focus, Agile, Scientific and Timeframe (COV-FAST)</b></p>
    <p style = "font-family:Verdana; font-size:14px">There are methodologies to be considered logistic regression, random forest or neural networking etc. We can use the same preprocessing dataset and try each options methodologies in seperate notebooks</p>
</div>
<h3>Methodologies Overview</h3>
<h3>Data Analysis PACE Steps:</h3>
   <ol style="font-family:Verdana; font-size:16px">
    <li><img style="float:left" src="https://i.imgur.com/gIne5bH.png" width="50"> Define (Plan & Analyze - EDA) - PART 1</li> 
    <blockquote>
    <ol>Understand your data in the problem context
        <br>EDA - check model, assumption & select model
    </ol>
    </blockquote>
        <li><img style="float:left" src="https://i.imgur.com/rb8V6X5.png" width="50">Measure (Analyze - EDA)</li>
    <blockquote>
    <ol> EDA - check model, assumption & select model
     </ol>
    </blockquote>
    <li><img style="float:left" src="https://i.imgur.com/J4M3HKM.png" width="50">Analyze (Construct) </li>
    <blockquote>
    <ol>Contruct and evaluate model
     </ol>
    </blockquote>
    <li><img style="float:left" src="https://i.imgur.com/wpcEXQC.png" width="50">Improve (Execute) - interpret model and share the story</li>
    <br>
    <li>Control</li> 
   
 <h3 style = "text-align:center">Table 1.1. PACE Methodologies (Plan)</h3>
<table style="color:black;
           display:fill;
           border-colapse: colapse;
           width: 100%;
           border: 1px solid black;
           border-collapse: collapse;
           border-style: solid;
           border-radius:5px;
           background-color:#5642C5;
           font-size:110%;
           font-family:Verdana;
           letter-spacing:0.5px">
  <tr>
    <th colspan="5">PACE 4 Steps</th>
  </tr>
    <tr>
        <th colspan="2"><em>Define<br>Plan</em></th>
    <th colspan="3">Customer expectations of the process? <br>Understand your data in the problem context</th>
  </tr>
  <tr>
    <th>Steps</th>
    <th>Description</th>
    <th>Focus</th>
    <th>Tools</th>
    <th>Timeline</th>
  </tr>
  <tr>
    <td style ="text-align:left" >A
      </td>
    <td>Identify Project's CTQ</td>
    <td style = "text-align:center">      
      Y 
    </td>
    <td style ="text-align:left">
      CTQ drill down tree, 
        VOC, Pareto, Bar Chart 
    </td>
      <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
  <tr>
     <tr>
    <td style ="text-align:left" >B
      </td>
    <td>Define Process Mapping</td>
    <td style = "text-align:center">      
     Y
    </td>
    <td style ="text-align:left">
     SIPOC
    </td>
    <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
    <td style ="text-align:left" >C
      </td>
    <td>Establish The Team's Charter</td>
    <td style = "text-align:center">      
     Y
    </td>
    <td style ="text-align:left">
     Milestones, Team Members, Stakeholders, Project Charter 
    </td>
    <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
    <tr>
<tr style = "border-top: solid ">
    <th colspan="5">Data Analytics 6 Steps - Project Initiation and Planning Phase</th>
  </tr>
    <tr>
        <th colspan="2"><em>Ask and Prepare</em></th>
    <th colspan="3">Customer expectations of the process? </th>
  </tr>
    
  <tr>
    <th>Steps</th>
    <th>Description</th>
    <th>Focus</th>
    <th>Tools</th>
    <th>Timeline</th>
  </tr>
  <tr>
    <td style ="text-align:left" >Ask - Mining
      </td>
    <td>A Clear Statement of The Business Task</td>
    <td style = "text-align:center">      
      Y 
    </td>
    <td style ="text-align:left">
      CTQ drill down tree, 
        VOC, Pareto, Bar Chart  
    </td>
    <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
  <tr>
    <td style ="text-align:left" >Ask - Mining
      </td>
    <td>Mapping Key Stakholders</td>
    <td style = "text-align:center">      
     Y
    </td>
    <td style ="text-align:left">
     RACI Matrix
    </td>
    <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
  <tr>
    <td style ="text-align:left" >Prepare
      </td>
    <td>A Description of All Data Used</td>
    <td style = "text-align:center">      
     Y
    </td>
    <td style ="text-align:left">
     Data Preparation or Gathering 
    </td>
    <td style ="text-align:center">
      3 - 5 days 
    </td>
  </tr>
 </table>    

## Define the Process
<hr>
<table style="color:black;
           display:fill;
           border-colapse: colapse;
           width: 100%;
           border: 1px solid black;
           border-collapse: collapse;
           border-style: solid;
           border-radius:5px;
           background-color:#5642C5;
           font-size:110%;
           font-family:Verdana;
           letter-spacing:0.5px">
  <h3 style = "text-align:center">Table 1.2. SIPOC Analysis Loan Process</h3>
  <tr>
    <th>Supplier (S)
    <th colspan="2">Input (I)</th>    
    <th colspan="4">Process (P)</th>
    <th colspan="2">Output (O)</th>
    <th colspan="2">Customer (C)</th>  
  </tr>
    <td>Lender
    <br>Account Manager
    <br>Relationship Manager
    <br>Apraisal</td>
    <td colspan="2">
        <ul>
            <li>Organize documents
            <li>Apply for loan
            <li>Customer management
            <li>Receive documents
            <li>Receive application
        </ul></td>
    <td style="background:LightSkyBlue;text-align:center">Loan application and document submission</td>
    <td style="background:LightSkyBlue;text-align:center">Screening process</td>
    <td style="background:LightSkyBlue;text-align:center">Negotiation</td>
    <td style="background:LightSkyBlue">Loan application finalization</td>
    <td style="background:LightSkyBlue">Approval of loan</td>
    <td colspan="2" style="margin: auto; display:fill; word-wrap: break-word">
        <ul>
            <li>Personal Loan
            <li>Corporate Loan
            <li>etc
        </ul>
    </td>
    <td colspan="2" style="margin: auto; display:fill; word-wrap: break-word">
        <ul>
            <li>Personal
            <li>Corporate
            <li>etc
        </ul>
    </td>
 </table>
<div class="alert alert-block alert-warning">
<img style="float: left" src="https://i.postimg.cc/kXz8cFqC/005-Img-Yellow-Notes-Draft-1-20220819.png" width="60">
<b style = "font-family: Arial; font-size: 16px">Note:</b><p style = "font-family:Verdana; font-size:14px">Discussion should be conducted with the process' owner</p>
</div>
  
## Stakeholder Analysis
<hr>
<span style ="font-family:Verdana; font-size:16px; text-align:justify">In this project we will only be mapping the stakeholders that were mentioned in articles or sources with high impacts and major role to the Loan project.</span>

<h3 style = "text-align:center">Table 1.3. Stakeholder Analysis</h3>
<table style="color:black;
           display:fill;
           border-colapse: colapse;
           width: 100%;
           border: 1px solid black;
           border-collapse: collapse;
           border-style: solid;
           border-radius:5px;
           background-color:#5642C5;
           font-size:110%;
           font-family:Verdana;
           letter-spacing:0.5px">
  
  <tr>
    <th>Stakeholders</th>
    <th>Role</th>
    <th colspan="2">Involvement</th>    
    <th>Power or Influence (H/M/L)</th>
    <th>Interest (H/M/L)</th>
    <th colspan="2">Engagement</th>  
  </tr>
  <tr>
    <td>Loan Director</td>
    <td style = "text-align:left">      
      Project sponsor 
    </td>
    <td colspan="2" style ="text-align:left">
    Makes high-level decisions; serves as team resource
    </td>
    <td style ="text-align:center">
      H 
    </td> 
    <td style ="text-align:center">
      L
    </td>
    <td colspan="2" style ="text-align:left">
      Communicate regularly, but not daily. Ask questions and give updates. 
    </td>
  </tr>
  <tr>
    <td>Dept Head</td>
    <td style = "text-align:left">      
      Project owner 
    </td>
    <td colspan="2" style ="text-align:left">
     <ul>
         <li>advisory role and providing valid information
         <li>Implementation of preventive, diagnosis and treatment measures
         <li>Allow re-export of surplus imported data or any required items to support project
         <li>Formulation of the business requirements
        <li>Formulation and implementation of equitable distribution of execution
        </ul> 
    </td>
    <td style ="text-align:center">
      M 
    </td> 
    <td style ="text-align:center">
      H 
    </td>
    <td colspan="2" style ="text-align:left">
      <ul>
        <li>Informing, mentoring and coaching
        <li>Monitoring the proper implementation of interventions
        <li>Coordination in informing
        <li>Official information reference in the loan Management and
control
        <li>Training and consulting with other related organizations and institutions
        </ul> 
    </td>
  </tr>
  <tr>
    <td>Appraisal Manager</td>
    <td style = "text-align:left">      
      Project leader 
    </td>
    <td colspan="2" style ="text-align:left">
     <ul>
      <li>Project-rules making and planning
     <li>Facilitate and synergy in inter-sectorial cooperation
     <li>Identifying problems and providing solutions in the form of executive
       approvals
     </ul> 
    </td>
    <td style ="text-align:center">
      M 
    </td> 
    <td style ="text-align:center">
      H 
    </td>
    <td colspan="2" style ="text-align:left">
       <ul>
      <li>Synergy and strengthening of various capabilities across the team and organization,
       directing and Mobilizing all capacities within the team.
      <li>Lead project from the start to clossing.
      </ul> 
    </td>
  </tr>
  
 </table>
  
  <h3 style = "text-align:center">Table 1.4. Project Charter</h3>
<table style="color:black;
           display:fill;
           border-colapse: colapse;
           width: 100%;
           border: 1px solid black;
           border-collapse: collapse;
           border-style: solid;
           border-radius:5px;
           background-color:#5642C5;
           font-size:110%;
           font-family:Verdana;
           letter-spacing:0.5px">
  
  <tr>
    <th colspan ="4" style="text-align:center">Project Loan Repaid</th>
  </tr>
    <tr>
    <th colspan ="4" style="text-align:center">30th July 2022</th>
  </tr>
  <tr>
      <th colspan ="4" style="text-align:center">Document Status: <del>Draft</del> | In Review | <del>Approved</del></th>
  </tr>
  <tr>
      <th colspan ="4" style="text-align:center">Executive Summary</th>
  </tr>
  <tr>
      <td colspan ="4" style="text-align:center">develop a model that predicts employee churn (stayed or left) profile.</td>
  </tr>
  <tr style ="background:LightSkyBlue;text-align:center">
      <td colspan ="2">Business Case</td>
      <td colspan ="2">Problem/Opportunity Statement</td>
  </tr>
 <tr style="text-align:left">
      <td colspan ="2">Based on the data average loan amount that people charged off is USD 15,000.00 with number people around 77,500 are equal to USD 1,162,500,000 of total loan amount being charged off. Moreover, The average interest rate is almost 16% which is approximately $ 186,000,000
</td>
      <td colspan ="2"><em>In this project</em> we establish a model to find ways to<b> predict profile that likely to churn </b>
<br><br>The goal of selective lender is to identify customer who are relevant to repaid the loan.
     </td>
  </tr>

 <tr style ="background:LightSkyBlue;text-align:center">
      <td colspan ="2">Goal Statement</td>
      <td colspan ="2">Deliverables (Key Results)</td>
 </tr>
 <tr style="text-align:left">
      <td colspan ="2">Primary metric:
     <ul>
         <li>% Loan status fully paid
     </ul>
     Secondary Metric:
     <ul>
         <li>Increase nett interest margin
         <li>Reduce bad debt expenses
     </ul>
     </td>
      <td colspan ="2">Primary Key Results:
     <ul>
         <li>Increase loan status fully paid - reduce 50% of charged off lenders (the data provide the employee who fully paid and Charged off).
     </ul>
     Secondary Key Results:
     <ul>
         <li>more than $90,000,000
         <li>USD
     </ul>
     </td>
  </tr>
 
 <tr style ="background:LightSkyBlue;text-align:center">
      <td colspan ="2">Benefits, Cost, and Budget</td>
      <td colspan ="2">Scope and Exclusion</td>
 </tr>
 <tr style="text-align:left">
      <td colspan ="2">Benefits:
     <ul>
         <li>Increse net margin
         <li>Reducing bad debt expenses
     </ul>
     Costs:
     <ul>
         <li>Training cost 
         <li>Lending cost 
     </ul>
     Budget Needed:
           <span>TBD</span>
     </td>
      <td colspan ="2">In-Scope:
     <ul>
         <li>All department
         <li>Full time
     </ul>
     Out-of-Scope:
     <ul>
         <li>With purpose as in data. Personal loan
         <li>Lean measure such as ,Fastest selection process etc.
     </ul>
     </td>
  </tr>   
 
 <tr style ="background:LightSkyBlue;text-align:center">
      <td colspan ="2">Project Team</td>
      <td colspan ="2">Measuring Success</td>
 </tr>
 <tr style="text-align:left">
      <td colspan ="2"> 
     <ul>
         <li>Sponsor: Laura, Director of HR
         <li>Owner: Washington, Operations Manager
         <li>Leader:Joko, Senior   
             Appraisal Manager
         <li>Member: Denzel, Data Analysis Manager, jennifer, Senior Data Analyst, Wahyu Senior Data Analyst   
     </ul>
     </td>
  
  <td colspan ="2">Deliverables after solutions implementation:
     <ul>
         <li>Increase loan being fully paid   
     </ul>
  </td>
  </tr>
  </table>
  
  <img style="float:left" src="https://i.imgur.com/wpcEXQC.png" width="50"><div style = "font-family: Arial; font-size: 16px">
    Execute</div>

<h3 style = "text-align:center">Figure 1. Feature Importance Loan Fully Paid or Charged Off Profile</h3>

<div style = "display:fill;
           border-colapse: colapse;
           width: 50%;
           margin: auto"> 
    
![keras_feature_shap](https://github.com/whyzie/Proj002-P01-DL-TensorFlow-Loan-20220730/assets/97485455/3dfa0207-6978-4b56-8157-16163b2fc232)
  
## Conclussion
<hr>

<div class="alert alert-block alert-success" style="font-family:verdana; font-size:14px">The tensor flow/keras model achieved precision of 90%, recall of 89%, f1-score of 87% (all weighted averages), and accuracy of 89%, on the test set.
<br><br>    
Precision and recall scores are both useful to evaluate the correct predictive capability of a model because they balance the false positives and false negatives inherent in prediction.
<br><br>
The model shows a precision score of 90%, suggesting the model is good at predicting true positives—meaning the loan will be fully paid-while balancing false positives. The recall score of 89% also shows quite good performance in predicting true negatives—where the loan will be charged off—while balancing false negatives.These two metrics combined can give a better assessment of model performance than accuracy does alone. 
<br><br>    
The feature importance graph seems to confirm that 9 certain area codes, interest rate, open account, 2(two) sub grades (B2, C2 and D4) and for rent and others are the "most important" features for this model. 
<br>    
The models and the feature importances extracted from the models confirm that the profile whofully paid loans in certain area and   . 
<br><br>
To acquired more fully paid lenders, the following recommendations could be presented to the stakeholders:
<br><br>
<ol>
<li> There are 9 area zipcode. Further investigation whether it is a certain area, office area, housing area etc.
<li>Offer low interest rate to people who have proven good record of paying their loan. We still need to elaborate further the range of the interest rate whether any particular rate.
<li>Should we limit open account,we need to see the range of the open account.
<li>Bank should always be prudent but we can be more linient to people with subgrade B2,C2 and D4 and loan for rent or certain purposes mentioned in feature importance.
<br><br>
Next Steps
<br><br>
It may be justified to still have some concern about data leakage. As this is unsupervised learning, the model still learning in which case need a strong base rate.  We can still elaborate more in the interest,open account,mortgage account, installment or drop the zipcode if further analysis on those area don't show a distinctive area.
</div>

  
     
  
