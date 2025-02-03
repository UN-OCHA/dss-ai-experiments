# ReliefWeb Training Tagging Accuracy Analysis - Multiple Vocabularies

This script analyzes the tagging accuracy for ReliefWeb training ads, comparing automated tagging using AWS Bedrock Titan Premier against historical manual tagging for both professional functions and themes.

## Purpose

The purpose of this script is to evaluate the accuracy of automated training ad tagging using AWS Bedrock Titan Premier. It processes training data and compares AI-generated tags for both professional functions and themes with the original human-assigned tags. This analysis helps assess the potential of automated tagging systems for future use in the ReliefWeb training posting workflow.

## Features

- Fetches training data, professional functions, and themes from the ReliefWeb API
- Processes training descriptions using AWS Bedrock Titan Premier
- Compares AI-generated professional functions and themes with original human-assigned tags
- Implements rate limiting for API requests
- Utilizes parallel processing for efficient training analysis
- Outputs results to a TSV file for further analysis

## Prerequisites

- Python 3.x
- AWS account with access to Bedrock
- ReliefWeb API access

## Configuration

The script requires a `config.json` file in the same directory with the following structure:

```json
{
  "AWS_ACCESS_KEY_ID": "your_aws_access_key",
  "AWS_SECRET_ACCESS_KEY": "your_aws_secret_key"
}
```

## Usage

1. Ensure all required Python libraries are installed
2. Place the `training-data.tsv` file in the same directory as the script
3. Run the script: `python script_name.py`

The script will process the training ads and output the results to `training_classification_results_titan_legacy.tsv`.

## Output

The script generates a TSV file with the following columns:

- Training ID
- Training URL
- Training Title
- Posted Date
- Editor Status
- Trusted Status
- Reviewed Status
- Actual Professional Functions
- Actual Themes
- AWS Bedrock Titan - Professional Functions
- AWS Bedrock Titan - Themes
- AWS Bedrock Titan - Reason
- AWS Bedrock Titan - Region
- AWS Bedrock Titan - Time
- AWS Bedrock Titan - Input Tokens
- AWS Bedrock Titan - Output Tokens

## Key Components

1. **Rate Limiting**: Implements a `RateLimiter` class to manage API request rates and token usage.
2. **Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent training processing.
3. **Prompt Generation**: Creates a detailed prompt for the AI model, including training details and classification instructions.
4. **Result Extraction**: Parses the AI model's response to extract professional functions, themes, and reasoning.
5. **Output Queue**: Manages the output of processed training ads to ensure ordered printing of results.

## Data Source

The script uses a dataset of 5000 legacy training postings from before 2021, retrieved from the ReliefWeb database using the following SQL query:

```sql
SELECT
  n.nid AS nid,
  DATE_FORMAT(FROM_UNIXTIME(n.created), '%Y-%m-%d') AS posted,
  IF(un.mail LIKE '%reliefweb.int', 'yes', 'no') AS editor,
  IF(SUM(IF(nfr.moderation_status = 'published' AND ur.mail NOT LIKE '%reliefweb.int', 1, 0)) > 0, 'yes', 'no') AS trusted,
  IF(SUM(IF(ur.mail LIKE '%reliefweb.int', 1, 0)) > 0, 'yes', 'no') AS reviewed
FROM node_field_data AS n
LEFT JOIN node_field_revision AS nfr
  ON nfr.nid = n.nid
LEFT JOIN node_revision AS nr
  ON nr.vid = nfr.vid
LEFT JOIN users_field_data AS ur
  ON ur.uid = nr.revision_uid
INNER JOIN users_field_data AS un
  ON un.uid = n.uid
WHERE
  n.type = 'training' AND
  n.created < UNIX_TIMESTAMP('2021-01-01 00:00:00') AND
  n.moderation_status IN ('published', 'expired')
GROUP BY n.nid
ORDER BY n.nid DESC
LIMIT 5000;
```

This query retrieves essential metadata for each training posting:

- `nid`: The unique training identifier
- `posted`: The date the training was posted
- `editor`: Indicates if the training was posted by a ReliefWeb editorial team member
- `trusted`: Indicates if the training was posted by a trusted submitter (not part of the editorial team)
- `reviewed`: Indicates if the training was reviewed by the editorial team

Using these training identifiers, the script then fetches additional content (such as the training title and description) from the ReliefWeb API to perform the tagging accuracy analysis.

## Notes

- The script is designed to handle large datasets efficiently, processing training ads in parallel.
- It includes error handling and logging for robust execution.
- The README provides a comprehensive overview of the script's functionality, setup, and usage.
- Unlike job postings, training ads can have multiple professional functions (up to 3) instead of a single career category.
