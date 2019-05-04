package com.atguigu.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

/**
  * Movie数据集，数据集字段通过分割
  *
  * 151^                          电影的ID
  * Rob Roy (1995)^               电影的名称
  * In the highlands ....^        电影的描述
  * 139 minutes^                  电影的时长
  * August 26, 1997^              电影的发行日期
  * 1995^                         电影的拍摄日期
  * English ^                     电影的语言
  * Action|Drama|Romance|War ^    电影的类型
  * Liam Neeson|Jessica Lange...  电影的演员
  * Michael Caton-Jones           电影的导演
  *
  * tag1|tag2|tag3|....           电影的Tag
  **/

case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String,
                 val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

/**
  * Rating数据集，用户对于电影的评分数据集，用，分割
  *
  * 1,           用户的ID
  * 31,          电影的ID
  * 2.5,         用户对于电影的评分
  * 1260759144   用户对于电影评分的时间
  */
case class MovieRating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)

/**
  * MongoDB的连接配置
  *
  * @param uri MongoDB的连接
  * @param db  MongoDB要操作数据库
  */
case class MongoConfig(val uri: String, val db: String)

//推荐
case class Recommendation(rid: Int, r: Double)

// 用户的推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

//电影的相似度
case class MovieRecs(uid: Int, recs: Seq[Recommendation])

object OfflineRecommender {
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val USER_MAX_RECOMMENDATION = 20

  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"
  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://bigdata:27017/recommender",
      "mongo.db" -> "reommender"
    )
    var sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores"))
      .set("spark.executor.memory", "10G").set("spark.driver.memory", "10G")
    //基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    //创建一个MongoDBConfig
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))
    //申明隐式参数
    import spark.implicits._
    //读取mongoDB中的业务数据
    val ratingRDD = spark.read //返回一个新的dataframe
      .option("uri", mongoConfig.uri) //连接参数
      .option("collection", MONGODB_RATING_COLLECTION) //连接参数
      .format("com.mongodb.spark.sql")
      .load() //加载为一个新的DataFrame
      .as[MovieRating].rdd //返回一个新的Dataset的rdd
      .map(r => (r.uid, r.mid, r.score)).cache() //映射到一个新的元祖上

    //用户数据集RDD[Int]，从评分数据集中拿到
    val userRDD = ratingRDD.map(m => m._1).distinct()
    //电影数据集RDD[Int]，从mongodb获取
    var movieRDD = spark.read.option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie].rdd
      .map(m => m.mid).cache() //cache之后性能提升
    //创建训练数据集

    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))

    val (rank, iterations, lambda) = (50, 5, 0.01)
    //训练ALS模型

    val model = ALS.train(trainData, rank, iterations, lambda)
    //计算用户推荐矩阵
//
//    //需要构造一个usersProducts  RDD[(Int,Int)]
//    val userMovies = userRDD.cartesian(movieRDD)
//    //预测函数,会出来一个评分结果表RDD[Rating]
//    val preRatings = model.predict(userMovies)
//
//    //输出 用戶:UserRecs:[(mid,score),(mid,scro)]类型的推荐数据
//    val userRecs = preRatings
//      .filter(m => m.rating > 0) //过滤掉评分为0的
//      .map(rating => (rating.user, (rating.product, rating.rating))) //映射为user,(mid,rating)
//      .groupByKey() //RDD[(K, Iterable[V])]
//      .map {
//      //排序，取前20个电影推荐
//      case (uid, recs) => UserRecs(uid, recs.toList.sortWith((m, n) => m._2 > n._2)
//        .take(USER_MAX_RECOMMENDATION).map(m => Recommendation(m._1, m._2)))
//    }.toDF()
//    //保存到数据库
//
//    userRecs.write
//      .option("uri", mongoConfig.uri)
//      .option("collection", USER_RECS)
//      .mode("overwrite")
//      .format("com.mongodb.spark.sql")
//      .save()

    //计算电影的相似度矩阵

    //获取电影的特征矩阵
    val movieFeatures = model.productFeatures.map { case (mid, freatures) =>
      (mid, new DoubleMatrix(freatures))
    }

    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter { case (a, b) => a._1 != b._1 }//过滤，笛卡尔积后，同一个电影不纳入统计
      .map { case (a, b) =>
        val simScore = this.consinSim(a._2, b._2)//计算两个特征矩阵的余弦相似度
        (a._1, (b._1, simScore))
      }.filter(_._2._2 > 0.6)//大于0.6的才纳入
      .groupByKey()
      .map { case (mid, items) =>
        MovieRecs(mid, items.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()

    movieRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection",MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    //关闭Spark
    spark.close()

  }

  //计算两个电影之间的余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }
}
