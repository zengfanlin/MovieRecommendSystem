package MovieLensALS

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation._

case class Movie(movieId: Int, title: String, genres: Seq[String])

case class User(userId: Int, gender: String, age: Int, occupation: Int, zip: String)


object MovieLensALSexample {

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]"

    )
    val PATH = "D:\\00-workspace\\01_java\\MovieRecommendSystem\\MovieLens\\src\\main\\scala\\"
    // 需要创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("MovieLensALS").setMaster("local[*]")
    //创建一个sparksession
    // 创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    //基于配置生成sc
    val sc = spark.sparkContext

    import spark.implicits._
    val sqlContext = spark.sqlContext
    //Ratings analyst
    val ratingText = spark.sparkContext.textFile(PATH + "ratings.dat")
    //    ratingText.first()
    //映射成Rating标准格式
    val ratingRDD = ratingText.map(parseRating).cache()

    //    println("Total number of ratings: " + ratingRDD.count())
    //    println("Total number of movies rated: " + ratingRDD.map(_.product).distinct().count())
    //    println("Total number of users who rated movies: " + ratingRDD.map(_.user).distinct().count())
    //    val ratingDF = ratingRDD.toDF()
    val movieDF = sc.textFile(PATH + "movies.dat").map(parseMovie).toDF()
    //        val userDF = sc.textFile(PATH + "users.dat").map(parseUser).toDF()
    //    //    ratingDF.printSchema()
    //    //    movieDF.printSchema()
    //    //    userDF.printSchema()
    //    ratingDF.registerTempTable("ratings")
    //    movieDF.registerTempTable("movies")
    //    userDF.registerTempTable("users")
    //    val result = sqlContext.sql(
    //      """select title,rmax,rmin,ucnt
    //from
    //(select product, max(rating) as rmax, min(rating) as rmin, count(distinct user) as ucnt
    //from ratings
    //group by product) ratingsCNT
    //join movies on product=movieId
    //order by ucnt desc""")
    //    result.show()
    //
    //    val mostActiveUser=sqlContext.sql("""select user, count(*) as cnt
    //from ratings group by user order by cnt desc limit 10""")
    //    mostActiveUser.show()
    //    val result=sqlContext.sql("""select distinct title, rating
    //from ratings join movies on movieId=product
    //where user=4169 and rating>4""")
    //    result.show()

    //ALS
    //分割训练集、测试集
    val splits = ratingRDD.randomSplit(Array(0.8, 0.2), 0L)
    val trainingSet = splits(0).cache()
    val testSet = splits(1).cache()
    //    trainingSet.count()
    //    testSet.count()
    val model = new ALS().setRank(20) //20个隐含因子
      .setIterations(10) //迭代10次
      .run(trainingSet)
    //Array[Rating]给上面的最活跃的用户4169推荐5部电影
    val recomForTopUser = model.recommendProducts(4169, 5) //.foreach(println)

    //    val movieTitle = movieDF.map(array => (array(0), array(1))).collect
    //    val recomResult = recomForTopUser.map(rating => (movieTitle(rating.product), rating.rating)).foreach(println)
    //
    //映射一个user,product 的 rdd
    val testUserProduct = testSet.map {
      case Rating(user, product, rating) => (user, product)
    }
    //预测只需要一个usersProducts: RDD[(Int, Int)]类型
    val testUserProductPredict = model.predict(testUserProduct)
    //    testUserProductPredict.take(10).mkString("\n")

    //按(user, product)作为key构建map
    val testSetPair = testSet.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val predictionsPair = testUserProductPredict.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    //    根据key（user,product）来join
    val joinTestPredict = testSetPair.join(predictionsPair)
    //映射出评估结果(user, product), (ratingT:真实值, ratingP:预测值)
    val ratingTP = joinTestPredict.map {
      case ((user, product), (ratingT, ratingP)) =>
        (ratingP, ratingT)
    }
    //建立回归评估模型
    val evalutor = new RegressionMetrics(ratingTP)
    println(evalutor.meanAbsoluteError) //平均误差
    println(evalutor.rootMeanSquaredError) //均方根误差

    spark.close()
  }

  //Define parse function
  def parseMovie(str: String): Movie = {
    val fields = str.split("::")
    assert(fields.size == 3)
    Movie(fields(0).toInt, fields(1).toString, Seq(fields(2)))
  }

  def parseUser(str: String): User = {
    val fields = str.split("::")
    assert(fields.size == 5)
    User(fields(0).toInt, fields(1).toString, fields(2).toInt, fields(3).toInt, fields(4).toString)
  }

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toInt)
  }
}
