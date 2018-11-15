  
# Logistic Regression 逻辑回归
## 1.二元逻辑回归   
(1)回归是解决变量之间的映射关系（x->y），而逻辑回归则通过sigmoid函数将映射值限定在(0,1)。sigmod函数如下：  
 
 <!-- ![sigmoid](./sigmod.png "sigmoid") -->
 

假设特征是x，参数xita，线性函数可以表示为：1.3，而逻辑回归则是在其基础上套上一个逻辑函数（sigmoid）：1.4 
  
  逻辑回归属于线性函数，具有线性决策边界（面）：
<!--![sigmoid](./sigmoid_line.png "sigmoid") -->
 
 对于二分类，分类结果只有两种：y=1 or y=0，另y=1的概率为：1.5，y=0概率则为：1.5 。 
根据数据X=(x1,x3,...,xn),Y=(y1,y2,...,yn)定义最大似然估计：1.6 ，目的是找到使得likelihood最大化的参数xita，因此对其取log（可以-log最小化，此处未-）：1.6，一阶gradient为：1.7，二阶梯度hessian为：1.8 
 
（2）最优化：ml.regression采用了L-BFGS(L2)和OWLQN(L1)
<div align=center>
  <img src="imgs/optimization.png" width="400" hegiht="200" div align=center /></div>
 
 
(3)为了减少过拟合，加入正则项，损失函数变为:1.9
## 2.多元逻辑回归  
 （1）多元回归类似softmax，类别概率定义为： <div align=center>
  <img src="imgs/multinomial_1.tiff" width="180" hegiht="100" div align=center /></div>  
  <div align=center>
  <img src="imgs/multinomial_2.tiff" width="180" hegiht="100" div align=center /></div> 
  
 &nbsp;&nbsp;&nbsp;&nbsp;二元逻辑回归中权重为向量，多元逻辑回归中权重beta为矩阵，相当于多个二元逻辑回归（每个类别/每行）: <div align=center>
  <img src="imgs/multinomial_2_2.png" width="180" hegiht="100" div align=center /></div>  
 
&nbsp;&nbsp;&nbsp;&nbsp;上述模型中的参数可以任意伸缩，即对于任意常数值，都可以被加到所有参数，而每个类别的概率值不发生变化：  <div align=center>
<img src="imgs/multinomial_3.tiff" width="400" hegiht="300" div align=center /></div>
   
（2）对于数据中的一个实例instance，损失函数为：<div align=center>
<img src="imgs/multinomial_4.tiff" width="300" hegiht="100" div align=center /></div> 
其中，<img src="imgs/multinomial_5.tiff" width="120" hegiht="100" div align=center /></div>
 
 不论SGD,LBFGS还是OWLQN最优化，都需要计算损失函数对参数的一阶导数： 
 <div align=center>
<img src="imgs/multinomial_6.tiff" width="400" hegiht="300" div align=center /></div> 
其中，w_i是样本权重（暂时忽略不管）， 而 I_{y=k}：因为对第k个类别的参数beta_k求导，因此只有当前样本的y=k，损失函数的最后一项才计算：

<div align=center>
<img src="imgs/multinomial_7.tiff" width="120" hegiht="120" div align=center /></div>
<div align=center>
<img src="imgs/multinomial_9.tiff" width="300" hegiht="300" div align=center /></div>
上述公式中，当max(margin)>0时会导致运算溢出，因此需要一些调整，首先损失函数等价变换：
<div align=center>
<img src="imgs/multinomial_8.tiff" width="500" hegiht="500" div align=center /></div>  
进而，multiplier则变成：
<div align=center>
<img src="imgs/multinomial_10.tiff" width="300" hegiht="300" div align=center /></div> 




## 3.实例 
 
## 4.代码分析  
### 4.1  整体流程
逻辑回归（mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala）的主要代码体现在run函数的 `val (coefficientMatrix, interceptVector, objectiveHistory) = {}` 代码块中。其中前部分初始化参数和计算summary（feature的均值和标准差等），之后则是关键部分：**损失函数costFun和最优化方法optimizer**：   
如果不使用L1正则化，则采用LBFGS优化，否则利用OWLQN算法优化（因为L1不保证处处可导），两者都属于拟牛顿法，可参考博客http://www.cnblogs.com/vivounicorn/archive/2012/06/25/2561071.html
    
        val regParamL1 = $(elasticNetParam) * $(regParam)
        val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

        val bcFeaturesStd = instances.context.broadcast(featuresStd)
        
        #损失函数后面定义，其中包括梯度计算。必须定义，breeze/optimize中（LBFGS和OWLQN）会用到
        val costFun = new LogisticCostFun(instances, numClasses, $(fitIntercept),
          $(standardization), bcFeaturesStd, regParamL2, multinomial = isMultinomial,
          $(aggregationDepth))

        val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
          val standardizationParam = $(standardization)
          def regParamL1Fun = (index: Int) => {
            // Remove the L1 penalization on the intercept
            val isIntercept = $(fitIntercept) && index >= numFeatures * numCoefficientSets
            if (isIntercept) {
              0.0
            } else {
              if (standardizationParam) {
                regParamL1
              } else {
                val featureIndex = index / numCoefficientSets
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                if (featuresStd(featureIndex) != 0.0) {
                  regParamL1 / featuresStd(featureIndex)
                } else {
                  0.0
                }
              }
            }
          }
          new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, regParamL1Fun, $(tol))
        }
        
        #此处省略 初始化等操作#
        
        #该LogisticRegression类继承了FirstOrderMinimizer，因此使用FirstOrderMinimizer类中的iterations方法
        val states = optimizer.iterations(new CachedDiffFunction(costFun),
          new BDV[Double](initialCoefWithInterceptMatrix.toArray))

        /*
           Note that in Logistic Regression, the objective history (loss + regularization)
           is log-likelihood which is invariant under feature standardization. As a result,
           the objective history from optimizer is the same as the one in the original space.
         */
        val arrayBuilder = mutable.ArrayBuilder.make[Double]
        var state: optimizer.State = null
        #更新到最后一个迭代为最终值
        while (states.hasNext) {
          state = states.next()
          arrayBuilder += state.adjustedValue
        }
        bcFeaturesStd.destroy(blocking = false)
 其中states表示状态迭代器，每个迭代进行更新，state类在breeze/optimize/FirstOrderMinimizer.scala中，包括梯度，损失值等信息：
  
    case class State[+T, +ConvergenceInfo, +History](
       x: T,
      value: Double,
      grad: T,
      adjustedValue: Double,
      adjustedGradient: T,
      iter: Int,
      initialAdjVal: Double,
      history: History,
      convergenceInfo: ConvergenceInfo,
      searchFailed: Boolean = false,
      var convergenceReason: Option[ConvergenceReason] = None) {}

### 4.2  损失函数类 LogisticCostFun  
 &nbsp;&nbsp;&nbsp;&nbsp; 作用：在FirstOrderMinimizer的iterations中更新states中使用calculateObjective方法，其中调用DiffFunction.calculate。而LogisticCostFun则继承DiffFunction并重写calculate方法：计算loss 和 gradient with L2 regularization  <br>
  
```

/**
 * LogisticCostFun implements Breeze's DiffFunction[T] for a multinomial (softmax) logistic loss
 * function, as used in multi-class classification (it is also used in binary logistic regression).
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
private class LogisticCostFun(
    instances: RDD[Instance],
    numClasses: Int,
    fitIntercept: Boolean,
    standardization: Boolean,
    bcFeaturesStd: Broadcast[Array[Double]],
    regParamL2: Double,
    multinomial: Boolean,
    aggregationDepth: Int) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)
    val bcCoeffs = instances.context.broadcast(coeffs)
    val featuresStd = bcFeaturesStd.value
    val numFeatures = featuresStd.length
    val numCoefficientSets = if (multinomial) numClasses else 1
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures

    #利用logisticAggregator类分别计算loss和gradient，之后treeAggregate进行add和merge
    val logisticAggregator = {
      val seqOp = (c: LogisticAggregator, instance: Instance) => c.add(instance)
      val combOp = (c1: LogisticAggregator, c2: LogisticAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new LogisticAggregator(bcCoeffs, bcFeaturesStd, numClasses, fitIntercept,
          multinomial)
      )(seqOp, combOp, aggregationDepth)
    }

    val totalGradientMatrix = logisticAggregator.gradient
    val coefMatrix = new DenseMatrix(numCoefficientSets, numFeaturesPlusIntercept, coeffs.toArray)
    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coefMatrix.foreachActive { case (classIndex, featureIndex, value) =>
        // We do not apply regularization to the intercepts
        val isIntercept = fitIntercept && (featureIndex == numFeatures)
        #将L2正则项加入loss和相应梯度
        if (!isIntercept) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            if (standardization) {
              val gradValue = totalGradientMatrix(classIndex, featureIndex)
              totalGradientMatrix.update(classIndex, featureIndex, gradValue + regParamL2 * value)
              value * value
            } else {
              if (featuresStd(featureIndex) != 0.0) {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                val temp = value / (featuresStd(featureIndex) * featuresStd(featureIndex))
                val gradValue = totalGradientMatrix(classIndex, featureIndex)
                totalGradientMatrix.update(classIndex, featureIndex, gradValue + regParamL2 * temp)
                value * temp
              } else {
                0.0
              }
            }
          }
        }
      }
      0.5 * regParamL2 * sum
    }
    bcCoeffs.destroy(blocking = false)

    (logisticAggregator.loss + regVal, new BDV(totalGradientMatrix.toArray))
  }

```   
<br>
上述代码主要功能：（1）计算loss和gradient并且合并；（2）计算L2正则项加入loss和相应gradient；（3）对数据进行standardize。<br>
（1）loss和gradient： 。<br>


<br>

   类LogisticAggregator中，<br>包括binaryUpdateInPlace和multinomialUpdateInPlace