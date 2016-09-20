# Front Ends and Extensions for Hadoop

[Front Ends and Extensions Take Hadoop in New Directions](https://www.linux.com/news/front-ends-and-extensions-take-hadoop-new-directions) describe the extenders and connectors for Hadoop and examples of how Hadoop can be taken in new directions with these tools:

  - [AtScale](http://atscale.com/) makes data stored in Hadoop's file system accessible within popular Business Intelligence (BI) applications.
  - [Microsoft](https://azure.microsoft.com/en-us/documentation/articles/hdinsight-connect-excel-power-query/) is making it easier to work with Hadoop directly from the Excel spreadsheet. Hortonworks has also made a [straightforward tutorial](http://hortonworks.com/hadoop-tutorial/partner-tutorial-microsoft/) on how you can use Excel as a front end for culling insights with Hadoop.
  - [Talend Open Studio for Big Data](https://www.talend.com/landing-download-ppc/big-data?device=c&lang=en&utm_source=google&utm_medium=cpc&utm_campaign=NA%20Search%20-%20Branded%20-%20TOS&src=GoogleAdwordsOD_US&kid=null&utm_term=%2Btalend%20%2Bopen%20%2Bstudio&utm_content=talend%20open%20studio%20-%20epmb&utm_creative=85473310600&gclid=CO7gl-6zt84CFU-Bfgods60AGA) provides a friendly front end for easily working with Hadoop to mine large data sets.
  - [Twill](http://twill.apache.org/) is an abstraction over Apache Hadoop YARN that reduces the complexity of developing distributed Hadoop applications, allowing developers to focus more on their application logic.
  - [Kylin](http://kylin.apache.org/) is an open source Distributed Analytics Engine designed to provide an SQL interface and multi-dimensional analysis (OLAP) on Apache Hadoop, supporting extremely large datasets.
  - [Lens](https://lens.apache.org/) is a Unified Analytics platform. It provides an optimal execution environment for analytical queries in the unified view.

# Big Data System

[Qunar real-time stream processing systems](http://www.techrepublic.com/article/how-a-big-data-hack-brought-a-300x-performance-bump-and-killed-a-major-bottleneck/) uses Apache Mesos for cluster management. We use Mesos to manage Apache Spark, Flink, Logstash, and Kibana. Logs come from multiple sources and we consolidate them with Kafka. The main computing frameworks, Spark streaming and Flink, subscribe to the data in Kafka, process the data, and then persist the results to HDFS (Hadoop Distributed File System).
