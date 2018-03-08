# TwitterBot

Setup Instructions

1. Set up a database file to store the tweets
```
cd setup
./setup-sqlite.py ../data/extract.db
cd..
```

2. Set up Solr - https://www.howtoforge.com/tutorial/how-to-install-and-configure-solr-on-ubuntu-1604/ 

3. Create a Solr Core 
```
sudo su - solr -c "/opt/solr/bin/solr create -c twitter"
```

4. Setup the core
```
cd setup
 ./setup-solr.py ../conf/solr-setup.xml -s http://localhost:8983/solr/twitter/
cd ..
```

5. If you wish to delete the documents from a core, you may do the same by running the following link on your web browser - 
'''
localhost:8983/solr/test5/update?commit=true&stream.body=%3Cdelete%3E%3Cquery%3E*:*%3C/query%3E%3C/delete%3E
'''
