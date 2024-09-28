Here's a README.md file for the attached script:

# ReliefWeb Job Tagging Accuracy Analysis

This script analyzes the tagging accuracy for ReliefWeb jobs posted before 2021, comparing automated tagging using AWS Bedrock Titan Premier against historical manual tagging.

## Purpose

The purpose of this script is to evaluate the accuracy of automated job tagging using AWS Bedrock Titan Premier. It focuses on legacy job data from before 2021, a period when the ReliefWeb job team had more capacity to review submissions manually. By comparing the AI-generated tags with the original human-assigned tags, we can assess the potential of automated tagging systems for future use.

## Features

- Fetches job data and career categories from the ReliefWeb API
- Processes job descriptions using AWS Bedrock Titan Premier
- Compares AI-generated career categories with original human-assigned categories
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
2. Place the `job-data.tsv` file in the same directory as the script
3. Run the script: `python script_name.py`

The script will process the jobs and output the results to `job_classification_results_titan_legacy.tsv`.

## Output

The script generates a TSV file with the following columns:

- Job ID
- Job URL
- Job Title
- Posted Date
- Editor Status
- Trusted Status
- Reviewed Status
- Actual Job Category
- AWS Bedrock Titan - Category
- AWS Bedrock Titan - Reason
- AWS Bedrock Titan - Region
- AWS Bedrock Titan - Time
- AWS Bedrock Titan - Input Tokens
- AWS Bedrock Titan - Output Tokens

## Data Source

The script uses a dataset of 5000 legacy job postings from before 2021, retrieved from the ReliefWeb database using the following SQL query:

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
  n.type = 'job' AND
  n.created < UNIX_TIMESTAMP('2021-01-01 00:00:00') AND
  n.moderation_status IN ('published', 'expired')
GROUP BY n.nid
ORDER BY n.nid DESC
LIMIT 5000;
```

This query retrieves essential metadata for each job posting:

- `nid`: The unique job identifier
- `posted`: The date the job was posted
- `editor`: Indicates if the job was posted by a ReliefWeb editorial team member
- `trusted`: Indicates if the job was posted by a trusted submitter (not part of the editorial team)
- `reviewed`: Indicates if the job was reviewed by the editorial team

Using these job identifiers, the script then fetches additional content (such as the job title and description) from the ReliefWeb API to perform the tagging accuracy analysis.

## Notes

- The script uses the ReliefWeb API to fetch current career categories and job details
- AWS Bedrock Titan Premier is used for generating career category predictions
- The script is designed to handle large datasets efficiently, processing jobs in batches
