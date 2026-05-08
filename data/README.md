# Demo Incident Datasets

This folder contains curated sample incident exports used to demonstrate the product across multiple business contexts.

| File | Format | Scenario |
| --- | --- | --- |
| `sample_incidents_org1.csv` | CSV | Retail / E-commerce |
| `sample_incidents_org2.xlsx` | Excel | Healthcare / Hospital |
| `sample_incidents_org3.csv` | CSV | B2B SaaS Platform |

All files preserve the same schema required by the application:

- `incident_id`
- `opened_at`
- `closed_at`
- `priority`
- `state`
- `assignment_group`
- `assigned_to`
- `opened_by_email`
- `stakeholder_emails`
- `team_emails`
- `service`
- `category`
- `short_description`
- `description`
- `business_impact`
- `resolution`
- `location`
- `reopened_count`

The datasets are intentionally compact to keep AI prompts efficient while still showing differentiated business impact, recurring services, stakeholder context, and communication opportunities.
