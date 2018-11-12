

Within psql try


```sql
\dt Table
explain analyze select * from Table where whatever > .5
```
