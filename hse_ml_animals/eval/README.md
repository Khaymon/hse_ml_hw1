## Experiments
|                                                           model                                                           | CV f1 avg |  LB f1 |
|:-------------------------------------------------------------------------------------------------------------------------:|:---------:|:------:|
|                                                   gbm 50 Breed features                                                   |   0.4014  | 0.3974 |
|                                                   gbm all Breed features                                                  |   0.4039  |        |
|                                          gbm all Breed features blending 5 splits                                         |   0.4026  |        |
|                                   logreg normalized all Breed features blending 5 splits                                  |   0.3839  |        |
|                                             knn 5 neighbors all Breed features                                            |   0.3320  |        |
|                                         gbm all Breed features, datetime features                                         |   0.4122  | 0.3873 |
|                                  gbm all Breed features, datetime features, name feature                                  |   0.4304  | 0.4153 |
|                                gbm min_df CountVectorizer, datetime features, name feature                                |   0.4311  | 0.4137 |
|                          gbm min_df CountVectorizer, datetime features, name feature, poly deg 2                          |   0.4408  | 0.4135 |
|               gbm min_df CountVectorizer, datetime features with hours and minutes, name feature, poly deg 2              |   0.4803  |        |
|               gbm min_df CountVectorizer, datetime features with hours and minutes, name length, poly deg 2               |   0.4860  |        |
|         gbm min_df CountVectorizer, datetime features with hours and minutes, name length, num colors, poly deg 2         |   0.4809  |        |
| gbm min_df CountVectorizer, datetime features with hours and minutes and day of year, name length, num colors, poly deg 2 |   0.4708  |        |
|   hist gbm, weighted classes, min_df CountVectorizer, datetime features with hours and minutes, name length, poly deg 2   |   0.5021  | 0.4807 |
|                                                                                                                           |           |        |
