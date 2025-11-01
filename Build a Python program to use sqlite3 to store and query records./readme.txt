README - How to Run and Test the Flask Model wrapper API

1. Install dependencies
   Make sure Python 3 and pip are installed.
   Then right click on the blank space of this folder and click open in terminal and type without quotes:

   "pip install -r requirements.txt"
   (skip the 1st step if you already done it)

2. to Start the Flask server
   Run this command without quotes:
   "python app.py"
   If everything’s fine, you’ll see:

* Running on [http://127.0.0.1:5000] or (http://127.0.0.1:5000)

3. Check if the server is healthy
   Open this link in your browser without quotes:
   after "http://127.0.0.1:5000" type "/health" and 
   it should look like this: "http://127.0.0.1:5000/health"
   It should return:
   {"status": "ok", "model_loaded": true}

4. View model metadata
   To see info about the model:
   [http://127.0.0.1:5000/metadata]

5. Make a prediction
   Use Postman, curl, or any REST client.

Example (command line) without quotes:
"curl -X POST [http://127.0.0.1:5000/predict] -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'"

or if you are using powershell/cmd:
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"features": [5.1, 3.5, 1.4, 0.2]}'

Expected output:
{"prediction": 0, "label": "setosa", "probabilities": [[0.98, 0.02, 0.0]], "model_version": "1.0.0"}

(or)

label  model_version prediction probabilities
-----  ------------- ---------- -------------
setosa 1.0.0                  0 {1.0 0.0 0.0}

6. Test batch predictions
   Send multiple inputs:
   curl -X POST [http://127.0.0.1:5000/predict_batch](http://127.0.0.1:5000/predict_batch) -H "Content-Type: application/json" -d '{"instances": [[5.1,3.5,1.4,0.2],[6.2,3.4,5.4,2.3]]}'

   Powershell version:
   Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict_batch" `
-Method Post `
-ContentType "application/json" `
-Body '{"instances": [[5.1,3.5,1.4,0.2], [6.2,3.4,5.4,2.3]]}'


Output example:
{"predictions": [0, 2], "labels": ["setosa", "virginica"]}

or

labels              model_version predictions probabilities
------              ------------- ----------- -------------
{setosa, virginica} 1.0.0         {0, 2}      {1.0 0.0 0.0, 0.0 0.0 1.0}

7. Stop the server
   Press Ctrl + C in the terminal.
