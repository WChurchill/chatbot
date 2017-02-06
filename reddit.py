import praw
import config
# client id and client secret are defined in 'config'
# this file is excluded from the public repo
#
# if you want to scrape reddit yourself, register your project at
# https://www.reddit.com/wiki/api

# the three essential parameters for a read-only reddit instance
reddit = praw.Reddit(user_agent='linux:chatbot:v0.1 (by /u/lispguru)',
                     client_id=config.client_id,
                     client_secret=config.client_secret)

print(reddit.read_only)
