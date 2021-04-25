#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Mat oneHotEncode(const cv::Mat& labels, const int numClasses)
{
    cv::Mat output(labels.rows, numClasses, CV_32F, 0.0f); // all zero, initially

    const int* labelsPtr = labels.ptr<int>(0);

    for (int i=0; i<labels.rows; i++)
    {
        float* outputPtr = output.ptr<float>(i);
        int id = labelsPtr[i];
        outputPtr[id] = 1.f;
    }

    return output;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainMLP(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto mlp = cv::ml::ANN_MLP::create();

    int nFeatures = dataset->getNVars();
    int nClasses = 10;

    cv::Mat_<int> layers(4,1);
    layers(0) = nFeatures;     // input
    layers(1) = nClasses * 32; // hidden
    layers(2) = nClasses * 16; // hidden
    layers(3) = nClasses;      // output,

    mlp->setLayerSizes(layers);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0, 0);
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 500, 0.0001));
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

    cv::Mat trainData = dataset->getTrainSamples();
    cv::Mat trainLabels = dataset->getTrainResponses();
    trainLabels = oneHotEncode(trainLabels, nClasses);

    mlp->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);

    mlp->save("MLP.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testMLP(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto mlp = cv::ml::ANN_MLP::load("MLP.xml");

    auto testSamples = dataset->getTestSamples();
    auto testResponses = dataset->getTestResponses();

    cv::Mat result;
    mlp->predict(testSamples, result);

    float error = 0.f;
    for(int i=0; i<result.rows; ++i)
    {
        double minVal, maxVal;
        int minIdx, maxIdx;
        cv::minMaxIdx(result.row(i).t(), &minVal, &maxVal, &minIdx, &maxIdx);
        int prediction = maxIdx;
        int testResponse = testResponses.at<int>(i);

        if(prediction != testResponse)
            error += 1.f;
    }

    error /= testSamples.rows;

    return error*100;
}
