## Step 0 ##
**Before all the experiments about PAEval, you should check if you have the access to ChatGPT-4o api.**

## Step 1 ##
After you get the answers from VLMs, you should make a **descriptions** and an **explanations** json files for every method, and put them into ```PAEval/results_from_VLM/``` (Except LAVAD in our experiment, because it only gives descriptions.)

They should look like below:
```bash
{
    "zipper": {
        "anomaly_free": [
            describ1/explain1,
            describ2/explain2,
            ...
        ]
        "stuck":[
            describ1/explain1,
            describ2/explain2,
            ...
        ]
        ...
    }
    "car": {
        "anomaly_free": [
            describ1/explain1,
            describ2/explain2,
            ...
        ]
        "stuck":[
            describ1/explain1,
            describ2/explain2,
            ...
        ]
        ...
    }
    ...
}
```

## Step 2 ##
Now you should have 2 json files in ```PAEval/results_from_VLM/```, and the scores can be calculate by running ```text_compare.py``` as below:
```bash
python text_compare.py
```
remember to change the 2 file paths. You can find the result in ```raw_output``` in json form.

**Note**:
1. In this step we use ChatGPT-4o to do the compare task. 
2. If you want to test only one object or one task (describ or explain), you can spicify the related args in the python file. By default the file will run both tasks.

## Step 3 ##
Now you need to calculate the average performances of the different abnormlities.
```bash
python evaluator.py
```
and you will find the new json results in ```scores```

## Step 4 ##
The final step is to generate the final performance of each method.
```bash
python avg.py
```
and you will find the results in ```result.txt```

## Special step ##
**LAVAD** is a little different. There are 2 points of the method:
1. LAVAD generate descriptions on every several frames. In our experiment we use ChatGPT-4o to **summarize a description** of the whole video from the descriptions of the frames first, and then start from the step 1.
2. LAVAD has only descriptions but no explanations, so you may need to specify the task in step 2, and the final score of its 'explain' should be 0.0.
