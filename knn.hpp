#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainKnn(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto k_nearest = cv::ml::KNearest::create();

    k_nearest->setDefaultK(5);
    k_nearest->setIsClassifier(true);

    k_nearest->train(dataset);

    k_nearest->save("KNN.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testKnn(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto k_nearest = cv::ml::KNearest::load("KNN.xml");

    std::vector<int32_t> predictions;
    auto error = k_nearest->calcError(dataset, true, predictions);

    return error;
}
