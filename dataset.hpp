#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<cv::Mat> extractDigits(const cv::Mat &img)
{
    const int digitSize = 20;
    std::vector<cv::Mat> digits;
    digits.reserve(5000);

    for(int i = 0; i < img.rows; i += digitSize)
    {
        for(int j = 0; j < img.cols; j += digitSize)
        {
            cv::Rect roi = cv::Rect(j,i,digitSize, digitSize);
            cv::Mat digitImg = img(roi);
            digits.push_back(digitImg);
        }
    }

    return digits;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void deskewDigits(std::vector<cv::Mat> &digits)
{
    for (auto& digit : digits)
    {
        cv::Moments m = cv::moments(digit);

        if(abs(m.mu02) < 1e-2) return;

        float skew = m.mu11/m.mu02;
        cv::Mat warpMat = (cv::Mat_<float>(2,3) << 1, skew, -0.5*digit.rows*skew, 0, 1, 0);
        cv::warpAffine(digit, digit, warpMat, digit.size(), cv::WARP_INVERSE_MAP|cv::INTER_LINEAR);
    }
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Mat extractFeatures(const std::vector<cv::Mat> &digits)
{
    cv::HOGDescriptor hog(cv::Size(20,20), cv::Size(8,8), cv::Size(4,4), cv::Size(8,8), 9, 1, -1,  cv::HOGDescriptor::HistogramNormType::L2Hys, 0.2, 0, 64, 1);

    const int featueSize = hog.getDescriptorSize();

    cv::Mat features(digits.size(), featueSize, CV_32FC1);

    for (int i=0; i<digits.size(); i++)
    {
        const auto &digitImg = digits[i];
        std::vector<float> descriptor;
        hog.compute(digitImg, descriptor);
        float* featuresPtr = features.ptr<float>(i);
        memcpy(featuresPtr, descriptor.data(), featueSize*sizeof(float));
    }

    return features;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Mat loadLabels()
{
    cv::Mat labels(5000, 1, CV_32SC1);
    int32_t* labelsPtr = labels.ptr<int32_t>(0);

    for (int i=0; i<5000; i++)
        labelsPtr[i] = i/500;

    return labels;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Ptr<cv::ml::TrainData> createTrainData(const std::string &imgPath)
{
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);

    auto digitImages = extractDigits(img);
    deskewDigits(digitImages);
    auto features = extractFeatures(digitImages);
    auto labels = loadLabels();

    cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);

    return data;
}
