# Big Data System

[Qunar real-time stream processing systems](http://www.techrepublic.com/article/how-a-big-data-hack-brought-a-300x-performance-bump-and-killed-a-major-bottleneck/) uses Apache Mesos for cluster management. We use Mesos to manage Apache Spark, Flink, Logstash, and Kibana. Logs come from multiple sources and we consolidate them with Kafka. The main computing frameworks, Spark streaming and Flink, subscribe to the data in Kafka, process the data, and then persist the results to HDFS (Hadoop Distributed File System).
