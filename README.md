# TwitterBot

Setup Instructions

0. Set up a database file to store the tweets
```
cd setup
./setup-sqlite.py ../data/extract.db
cd..
```

1. Set up Solr - https://www.howtoforge.com/tutorial/how-to-install-and-configure-solr-on-ubuntu-1604/ 

2. Create a Solr Core 
```
sudo su - solr -c "/opt/solr/bin/solr create -c twitter"
```

3. Setup the core
```
cd setup
 ./setup-solr.py ../conf/solr-setup.xml -s http://localhost:8983/solr/twitter/
cd ..
```

5.
