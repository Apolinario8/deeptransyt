Usage Instructions
===================

To run DeepTransyt, use the following command:
    
    ``` 


    python DeepTransyt.py -i <input_file> -o <output_file> -m <model> -t <threshold> 
    
    ```	
        
        where:

        -i <input_file> : input file in csv format with the following columns: 'SMILES', 'ID', 'Activity'
        -o <output_file> : output file in csv format with the following columns: 'SMILES', 'ID', 'Activity', 'Prediction'


        -m <model> : model to be used. Options are: 'DeepTransyt' or 'DeepTransyt-2'



        -t <threshold> : threshold to be used to classify the compounds. Default is 0.5.



        Example:

        ```
        python DeepTransyt.py -i input.csv -o output.csv -m DeepTransyt -t 0.5
        ```

        The output file will contain the following columns: 'SMILES', 'ID', 'Activity', 'Prediction'

        The 'Activity' column is the true activity of the compound and the 'Prediction' column is the predicted activity of the compound.

        The 'Prediction' column will contain the probability of the compound being active. If the probability is greater than the threshold, the compound will be classified as active, otherwise it will be classified as inactive.

        The 'ID' column is optional and can be used to identify the compounds.

        The 'SMILES' column is mandatory and must contain the SMILES representation of the compounds.

        The 'Activity' column is mandatory and must contain the true activity of the compounds. The true activity must be a binary value (0 or 1).








