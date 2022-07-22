## Webscraping Linux Repos--Predicting the programming language.
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary 
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Initial Questions

>
>
>

#### Project Objectives
>
>
>
#### Data Dictionary
>
>
>
|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| accessed | 509409 non-null: datetime64 | Datetime stamp of when the record was created (access occured) |
| path | 509409 non-null: object | url path of what content was accessed |
| ip | 509409 non-null: object | ip address of user who accessed content |
| user_id | 509409 non-null: int64 | Unique (assumed) user id for user who accessed content|
| program_id | 509409 non-null: float64 | Program id (Web Dev 1 = 1, Web Dev 2 = 2, DS = 3; 4 was removed) |
| program_type | 509409 non-null: object | Derived from program_id (Web Development or Data Science) |
| cohort | 509409 non-null: object | What cohort(s) user_id is in; known or imputed in analysis df |
| start_date | 509409 non-null: datetime64 | Start date of given cohort |
| end_date | 509409 non-null: datetime64 | End date of give cohort |
| lesson | 509409 non-null: object | The endpoint of access, if a lesson, derived from 'path' |
| hour | 509409 non-null: int64 | The hour of access (24-hour clock), derived from 'accessed' |


#### Goals and our Why of this project
>
>
>
##### Plan
>
>
>
#### Initial Hypotheses
> - **Hypothesis 1 -**


> - **Hypothesis 2 -** 


> - **Hypothesis 3 -**


### Executive Summary - Conclusions & Next Steps



<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>


### Reproduce Our Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Access to CodeUp MySql server
- [ ] Have loaded all common DS libraries
- [ ] Download all helper function files [acquire.py, wrangle.py, explore.py]
- [ ] Scrap notebooks (if desired, to dive deeper)
- [ ] Run the final report
