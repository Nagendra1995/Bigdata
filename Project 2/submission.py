from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer,StopWordsRemover
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import col,udf,struct
from pyspark.sql import *


class Selector(Transformer):
    def __init__(self, outputCols=['id','features', 'label']):
        self.outputCols=outputCols
       
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.select(*self.outputCols)
def binary(a,b):
    if a+b == 0.0:
        return 0.0
    elif a+b ==1.0 and a==1.0:
        return 2.0
    elif a+b ==1.0 and b==1.0:
        return 1.0
    else:
        return 3.0

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")
    #stopwords=StopWordsRemover(inputCol="words",outputCol="Nostopwords")
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)
    label_maker = StringIndexer(inputCol = input_category_col, outputCol = output_label_col)
    selector = Selector(outputCols = ['id','features', 'label'])
    return Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])
def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    nb_0_pred=[]
    nb_1_pred=[]
    nb_2_pred=[]
    svm_0_pred=[]
    svm_1_pred=[]
    svm_2_pred=[]
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        nb_model_0 = nb_0.fit(c_train)
        nb_pred_0 = nb_model_0.transform(c_test)
        nb_0_pred.append(nb_pred_0)
        nb_model_1 = nb_1.fit(c_train)
        nb_pred_1 = nb_model_1.transform(c_test)
        nb_1_pred.append(nb_pred_1)
        nb_model_2 = nb_2.fit(c_train)
        nb_pred_2 = nb_model_2.transform(c_test)
        nb_2_pred.append(nb_pred_2)
        svm_model_0 = svm_0.fit(c_train)
        svm_pred_0 = svm_model_0.transform(c_test)
        svm_0_pred.append(svm_pred_0)
        svm_model_1 = svm_1.fit(c_train)
        svm_pred_1 = svm_model_1.transform(c_test)
        svm_1_pred.append(svm_pred_1)
        svm_model_2 = svm_2.fit(c_train)
        svm_pred_2 = svm_model_2.transform(c_test)
        svm_2_pred.append(svm_pred_2)
    df0 = nb_0_pred[0].union(nb_0_pred[1]).union(nb_0_pred[2]).union(nb_0_pred[3]).union(nb_0_pred[4])
    df1= nb_1_pred[0].union(nb_1_pred[1]).union(nb_1_pred[2]).union(nb_1_pred[3]).union(nb_1_pred[4])
    df2= nb_2_pred[0].union(nb_2_pred[1]).union(nb_2_pred[2]).union(nb_2_pred[3]).union(nb_2_pred[4])
    df3= svm_0_pred[0].union(svm_0_pred[1]).union(svm_0_pred[2]).union(svm_0_pred[3]).union(svm_0_pred[4])
    df4= svm_1_pred[0].union(svm_1_pred[1]).union(svm_1_pred[2]).union(svm_1_pred[3]).union(svm_1_pred[4])
    df5= svm_2_pred[0].union(svm_2_pred[1]).union(svm_2_pred[2]).union(svm_2_pred[3]).union(svm_2_pred[4])
    df0 = df0.alias('df0')
    df1 = df1.alias('df1')
    df2 = df2.alias('df2')
    df3 = df3.alias('df3')
    df4 = df4.alias('df4')
    df5 = df5.alias('df5')
    joined_df = df0.join(df1, col('df0.id') == col('df1.id'), 'left').join(df2, col('df0.id') == col('df2.id'), 'left') .join(df3, col('df0.id') == col('df3.id'), 'left') .join(df4, col('df0.id') == col('df4.id'), 'left') .join(df5, col('df0.id') == col('df5.id'), 'left').select('df0.id', 'df0.group', 'df0.features','df0.label' ,'df0.label_0' ,'df0.label_1', 'df0.label_2', 'df0.nb_pred_0', 'df1.nb_pred_1', 'df2.nb_pred_2', 'df3.svm_pred_0', 'df4.svm_pred_1' ,'df5.svm_pred_2')
    udffunc=udf(binary,DoubleType())
    joined_df=joined_df.withColumn("joint_pred_0",udffunc(col('nb_pred_0'),col('svm_pred_0')))
    joined_df=joined_df.withColumn("joint_pred_1",udffunc(col('nb_pred_1'),col('svm_pred_1')))
    joined_df=joined_df.withColumn("joint_pred_2",udffunc(col('nb_pred_2'),col('svm_pred_2')))
    return joined_df
   
   
   
   
   

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    #pipeline = Pipeline(stages=[base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model])
    #fitted_pipeline = pipeline.fit(test_df)
    a=base_features_pipeline_model.transform(test_df)
    b=gen_base_pred_pipeline_model.transform(a)
    udffunc=udf(binary,DoubleType())
    b=b.withColumn("joint_pred_0",udffunc(col('nb_pred_0'),col('svm_pred_0')))
    b=b.withColumn("joint_pred_1",udffunc(col('nb_pred_1'),col('svm_pred_1')))
    b=b.withColumn("joint_pred_2",udffunc(col('nb_pred_2'),col('svm_pred_2')))
    c=gen_meta_feature_pipeline_model.transform(b)
    svm_pred = meta_classifier.transform(c)
    
    return svm_pred