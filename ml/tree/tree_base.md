<h1 align="center"> tree base—spark中树模型的基础类总结</h1>

<h2>一、Split</h2>
<div>
一个特征存在多个split：判断goLeft 还是goRight
<code><pre>
/**
 * Interface for a "Split," which specifies a test made at a decision tree node
 * to choose the left or right path.
 */
sealed trait Split extends Serializable {

  /** Index of feature which this split tests */
  def featureIndex: Int

  /**
   * Return true (split to left) or false (split to right).
   * @param features  Vector of features (original values, not binned).
   */
  private[ml] def shouldGoLeft(features: Vector): Boolean

  /**
   * Return true (split to left) or false (split to right).
   * @param binnedFeature Binned feature value.
   * @param splits All splits for the given feature.
   */
  private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[Split]): Boolean

  /** Convert to old Split format */
  private[tree] def toOld: OldSplit
}
</pre></code>
<br>
连续特征的Split（featureIndex，threshold）：<=threshold则goleft，否则goright
<code><pre>
/**
 * Split which tests a continuous feature.
 * @param featureIndex  Index of the feature to test
 * @param threshold  If the feature value is less than or equal to this threshold, then the
 *                   split goes left. Otherwise, it goes right.
 */
class ContinuousSplit private[ml] (override val featureIndex: Int, val threshold: Double)
  extends Split {

  override private[ml] def shouldGoLeft(features: Vector): Boolean = {
    features(featureIndex) <= threshold
  }

  override private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[Split]): Boolean = {
    if (binnedFeature == splits.length) {
      // > last split, so split right
      false
    } else {
      val featureValueUpperBound = splits(binnedFeature).asInstanceOf[ContinuousSplit].threshold
      featureValueUpperBound <= threshold
    }
  }

  override def equals(o: Any): Boolean = {
    o match {
      case other: ContinuousSplit =>
        featureIndex == other.featureIndex && threshold == other.threshold
      case _ =>
        false
    }
  }

  override def hashCode(): Int = {
    val state = Seq(featureIndex, threshold)
    state.map(Objects.hashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override private[tree] def toOld: OldSplit = {
    OldSplit(featureIndex, threshold, OldFeatureType.Continuous, List.empty[Double])
  }
}
</pre></code>
<br>
离散特征Split（featureIndex，leftCatgories，numCategories）：特征值在leftCatgories数组中则goleft，否则goright
<code><pre>
/**
 * Split which tests a categorical feature.
 * @param featureIndex  Index of the feature to test
 * @param _leftCategories  If the feature value is in this set of categories, then the split goes
 *                         left. Otherwise, it goes right.
 * @param numCategories  Number of categories for this feature.
 */
class CategoricalSplit private[ml] (
    override val featureIndex: Int,
    _leftCategories: Array[Double],
    @Since("2.0.0") val numCategories: Int)
  extends Split {

  require(_leftCategories.forall(cat => 0 <= cat && cat < numCategories), "Invalid leftCategories" +
    s" (should be in range [0, $numCategories)): ${_leftCategories.mkString(",")}")

  /**
   * If true, then "categories" is the set of categories for splitting to the left, and vice versa.
   */
  private val isLeft: Boolean = _leftCategories.length <= numCategories / 2

  /** Set of categories determining the splitting rule, along with [[isLeft]]. */
  private val categories: Set[Double] = {
    if (isLeft) {
      _leftCategories.toSet
    } else {
      setComplement(_leftCategories.toSet)
    }
  }

  override private[ml] def shouldGoLeft(features: Vector): Boolean = {
    if (isLeft) {
      categories.contains(features(featureIndex))
    } else {
      !categories.contains(features(featureIndex))
    }
  }

  override private[tree] def shouldGoLeft(binnedFeature: Int, splits: Array[Split]): Boolean = {
    if (isLeft) {
      categories.contains(binnedFeature.toDouble)
    } else {
      !categories.contains(binnedFeature.toDouble)
    }
  }

  override def hashCode(): Int = {
    val state = Seq(featureIndex, isLeft, categories)
    state.map(Objects.hashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def equals(o: Any): Boolean = o match {
    case other: CategoricalSplit => featureIndex == other.featureIndex &&
      isLeft == other.isLeft && categories == other.categories
    case _ => false
  }

  override private[tree] def toOld: OldSplit = {
    val oldCats = if (isLeft) {
      categories
    } else {
      setComplement(categories)
    }
    OldSplit(featureIndex, threshold = 0.0, OldFeatureType.Categorical, oldCats.toList)
  }

  /** Get sorted categories which split to the left */
  def leftCategories: Array[Double] = {
    val cats = if (isLeft) categories else setComplement(categories)
    cats.toArray.sorted
  }

  /** Get sorted categories which split to the right */
  def rightCategories: Array[Double] = {
    val cats = if (isLeft) setComplement(categories) else categories
    cats.toArray.sorted
  }

  /** [0, numCategories) \ cats */
  private def setComplement(cats: Set[Double]): Set[Double] = {
    Range(0, numCategories).map(_.toDouble).filter(cat => !cats.contains(cat)).toSet
  }
}

</pre></code>
</div>
<h2>二、TreePoint</h2>
<div>
为了更好适合DecisionTree的计算，将LabeledPoint转化为TreePoint：label不变，将features转化为bin indices
<code><pre>
/**
 * Internal representation of LabeledPoint for DecisionTree.
 * This bins feature values based on a subsampled of data as follows:
 *  (a) Continuous features are binned into ranges.
 *  (b) Unordered categorical features are binned based on subsets of feature values.
 *      "Unordered categorical features" are categorical features with low arity used in
 *      multiclass classification.
 *  (c) Ordered categorical features are binned based on feature values.
 *      "Ordered categorical features" are categorical features with high arity,
 *      or any categorical feature used in regression or binary classification.
 *
 * @param label  Label from LabeledPoint
 * @param binnedFeatures  Binned feature values.
 *                        Same length as LabeledPoint.features, but values are bin indices.
 */
private[spark] class TreePoint(val label: Double, val binnedFeatures: Array[Int])
  extends Serializable {
}

private[spark] object TreePoint {

  /**
   * Convert an input dataset into its TreePoint representation,
   * binning feature values in preparation for DecisionTree training.
   * @param input     Input dataset.
   * @param splits    Splits for features, of size (numFeatures, numSplits).
   * @param metadata  Learning and dataset metadata
   * @return  TreePoint dataset representation
   */
  def convertToTreeRDD(
      input: RDD[LabeledPoint],
      splits: Array[Array[Split]],
      metadata: DecisionTreeMetadata): RDD[TreePoint] = {
    // Construct arrays for featureArity for efficiency in the inner loop.
    val featureArity: Array[Int] = new Array[Int](metadata.numFeatures)
    var featureIndex = 0
    while (featureIndex < metadata.numFeatures) {
      featureArity(featureIndex) = metadata.featureArity.getOrElse(featureIndex, 0)
      featureIndex += 1
    }
    val thresholds: Array[Array[Double]] = featureArity.zipWithIndex.map { case (arity, idx) =>
      if (arity == 0) {
        splits(idx).map(_.asInstanceOf[ContinuousSplit].threshold)
      } else {
        Array.empty[Double]
      }
    }
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, thresholds, featureArity)
    }
  }

  /**
   * Convert one LabeledPoint into its TreePoint representation.
   * @param thresholds  For each feature, split thresholds for continuous features,
   *                    empty for categorical features.
   * @param featureArity  Array indexed by feature, with value 0 for continuous and numCategories
   *                      for categorical features.
   */
  private def labeledPointToTreePoint(
      labeledPoint: LabeledPoint,
      thresholds: Array[Array[Double]],
      featureArity: Array[Int]): TreePoint = {
    val numFeatures = labeledPoint.features.size
    val arr = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      arr(featureIndex) =
        findBin(featureIndex, labeledPoint, featureArity(featureIndex), thresholds(featureIndex))
      featureIndex += 1
    }
    new TreePoint(labeledPoint.label, arr)
  }

  /**
   * Find discretized value for one (labeledPoint, feature).
   *
   * NOTE: We cannot use Bucketizer since it handles split thresholds differently than the old
   *       (mllib) tree API.  We want to maintain the same behavior as the old tree API.
   *
   * @param featureArity  0 for continuous features; number of categories for categorical features.
   */
  private def findBin(
      featureIndex: Int,
      labeledPoint: LabeledPoint,
      featureArity: Int,
      thresholds: Array[Double]): Int = {
    val featureValue = labeledPoint.features(featureIndex)

    if (featureArity == 0) {
      val idx = java.util.Arrays.binarySearch(thresholds, featureValue)
      if (idx >= 0) {
        idx
      } else {
        -idx - 1
      }
    } else {
      // Categorical feature bins are indexed by feature values.
      if (featureValue < 0 || featureValue >= featureArity) {
        throw new IllegalArgumentException(
          s"DecisionTree given invalid data:" +
            s" Feature $featureIndex is categorical with values in {0,...,${featureArity - 1}," +
            s" but a data point gives it value $featureValue.\n" +
            "  Bad data point: " + labeledPoint.toString)
      }
      featureValue.toInt
    }
  }
}

</pre></code>
</div>
<h2>三、BaggedPoint</h2>
<div>
针对bagging(例如random forest)的一种数据表示形式，bagging的每个分类器都利用样本的一个子集进行训练，因此需要对样本进行抽样，形成若干个样本自己，由此设计了这种表示形式
<code><pre>
/**
 * Internal representation of a datapoint which belongs to several subsamples of the same dataset,
 * particularly for bagging (e.g., for random forests).
 *
 * This holds one instance, as well as an array of weights which represent the (weighted)
 * number of times which this instance appears in each subsamplingRate.
 * E.g., (datum, [1, 0, 4]) indicates that there are 3 subsamples of the dataset and that
 * this datum has 1 copy, 0 copies, and 4 copies in the 3 subsamples, respectively.
 *
 * @param datum  Data instance
 * @param subsampleWeights  Weight of this instance in each subsampled dataset.
 *
 * TODO: This does not currently support (Double) weighted instances.  Once MLlib has weighted
 *       dataset support, update.  (We store subsampleWeights as Double for this future extension.)
 */
private[spark] class BaggedPoint[Datum](val datum: Datum, val subsampleWeights: Array[Double])
  extends Serializable

private[spark] object BaggedPoint {

  /**
   * Convert an input dataset into its BaggedPoint representation,
   * choosing subsamplingRate counts for each instance.
   * Each subsamplingRate has the same number of instances as the original dataset,
   * and is created by subsampling without replacement.
   * @param input Input dataset.
   * @param subsamplingRate Fraction of the training data used for learning decision tree.
   * @param numSubsamples Number of subsamples of this RDD to take.
   * @param withReplacement Sampling with/without replacement.
   * @param seed Random seed.
   * @return BaggedPoint dataset representation.
   */
  def convertToBaggedRDD[Datum] (
      input: RDD[Datum],
      subsamplingRate: Double,
      numSubsamples: Int,
      withReplacement: Boolean,
      seed: Long = Utils.random.nextLong()): RDD[BaggedPoint[Datum]] = {
    if (withReplacement) {
      convertToBaggedRDDSamplingWithReplacement(input, subsamplingRate, numSubsamples, seed)
    } else {
      if (numSubsamples == 1 && subsamplingRate == 1.0) {
        convertToBaggedRDDWithoutSampling(input)
      } else {
        convertToBaggedRDDSamplingWithoutReplacement(input, subsamplingRate, numSubsamples, seed)
      }
    }
  }

  private def convertToBaggedRDDSamplingWithoutReplacement[Datum] (
      input: RDD[Datum],
      subsamplingRate: Double,
      numSubsamples: Int,
      seed: Long): RDD[BaggedPoint[Datum]] = {
        
  }

  private def convertToBaggedRDDSamplingWithReplacement[Datum] (
      input: RDD[Datum],
      subsample: Double,
      numSubsamples: Int,
      seed: Long): RDD[BaggedPoint[Datum]] = {
        
  }

  private def convertToBaggedRDDWithoutSampling[Datum] (
      input: RDD[Datum]): RDD[BaggedPoint[Datum]] = {
  }

}

</pre></code>
</div>
