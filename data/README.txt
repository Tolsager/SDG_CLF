Hello!

the data is available as a csv file named "allSDGtweets.csv" located in 
~/OneDrive - Danmarks Tekniske Universitet/SDG classifier NLP/SDG_df/

all tweets are original tweets, that is retweets, quoted tweets, replies, and promoted tweets have been discarded (filtered out at the query phase)

the file is composed of 22 comma separated variables
id: is the unique identifier of the tweet that can be used to search the tweet on the platform or in the "raw" json files
created_at: is the date at which the tweet was posted
text: is the text of the tweet
lang: the language used in the tweet reported by Twitter

this is then followed by 17 logical variables labelled #sdg1 to #sdg17 identifying whether each hashtag is present in the tweet

category: categorical variable whether the tweet is from the #sdg harvest (SDG) or from the control harvest (CONTROL)

nclasses: counts how many sdg classes are present in the tweet (you might want to restrict the training set to those tweets containing one and only one sdg class
(about 377k tweets have one and only one class, 340k english ones)

######################
the control set:

to harvest the control set and try to maintain some language consistency (and also because NOT queries are not practically feasible), the harvest 
was limited to the users that tweeted about sdgs (users discovered in the SDG harvest). we first used users tweeting unfrequently adding 'enough' users
to reach 320k control tweets (262k english tweets). The query discarded all tweets including "sdg" or "sustainability".



#####################
note there may still be sustainability related tweets in the control set and some tweets that do not contain a sdg hashtag in the SDG set.
check nclasses to help out and QC the data set.






