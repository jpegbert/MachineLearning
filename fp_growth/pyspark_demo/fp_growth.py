from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("FP-Growth关联规则").getOrCreate()
df = spark.createDataFrame([
    (0, 'A,B,E'),
    (1, 'B,D'),
    (2, 'B,C'),
    (3, 'A,B,D'),
    (4, 'A,C'),
    (5, 'B,C'),
    (6, 'A,C'),
    (7, 'A,B,C,E'),
    (8, 'A,B,C')
], ["id", "items"])
df = df.withColumn("items", F.split(df.items, ","))
df.show()

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.5)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()

