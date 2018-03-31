#!sqlite3

.tables
# conversation              response                  tag                     
# conversation_association  statement                 tag_association
.width 5 25 10 5 40
.mode columns
.mode column
.headers on
SELECT * FROM response LIMIT 9;
# id     text                       created_at  occur  statement_text                          
# -----  -------------------------  ----------  -----  ----------------------------------------
# 1      What is AI?                2017-11-26  2      Artificial Intelligence is the branch of
# 2      What is AI?                2017-11-26  2      AI is the field of science which concern
# 3      Are you sentient?          2017-11-26  2      Sort of.                                
# 4      Are you sentient?          2017-11-26  2      By the strictest dictionary definition o
# 5      Are you sentient?          2017-11-26  2      Even though I'm a construct I do have a 
# 6      Are you sapient?           2017-11-26  2      In all probability, I am not.  I'm not t
# 7      Are you sapient?           2017-11-26  2      Do you think I am?                      
# 8      Are you sapient?           2017-11-26  2      How would you feel about me if I told yo
# 9      Are you sapient?           2017-11-26  24     No.  
