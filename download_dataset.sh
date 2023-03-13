#!/bin/bash
BUCKET_NAME="aidata-2022-01-004"
IMAGES="013.객체 인식용 한국형 비전 데이터/06.품질검증/1.Dataset/1.원천데이터"
BBOX="013.객체 인식용 한국형 비전 데이터/06.품질검증/1.Dataset/2.라벨링데이터"
CAPTION="015.이미지 설명문 추출 및 생성용 한국형 비전 데이터/06.품질검증/1.Dataset/2.라벨링데이터/temporary"
RELATION="016.객체 간 관계성 인지용 한국형 비전 데이터/06.품질검증/1.Dataset/2.라벨링데이터"

images_download()
{
    /usr/local/bin/aws --endpoint-url=https://kr.object.ncloudstorage.com s3 sync "s3://${BUCKET_NAME}/${IMAGES}" images
}

bbox_download()
{
    /usr/local/bin/aws --endpoint-url=https://kr.object.ncloudstorage.com s3 sync "s3://${BUCKET_NAME}/${BBOX}" bbox
}

caption_download()
{
    /usr/local/bin/aws --endpoint-url=https://kr.object.ncloudstorage.com s3 sync "s3://${BUCKET_NAME}/${CAPTION}" caption
}

relation_download()
{
    /usr/local/bin/aws --endpoint-url=https://kr.object.ncloudstorage.com s3 sync "s3://${BUCKET_NAME}/${RELATION}" relation
}

run()
{
    case "$1" in
    images)
        images_download
    ;;
    bbox)
        bbox_download
    ;;
    caption)
        caption_download
    ;;
    relation)
        relation_download
    ;;
    *)
        echo ""
        echo "Usage: ${0} {images|bbox|caption|relation}"
        echo ""
        echo "       images   : download images"
        echo "       bbox     : download bbox meta data"
        echo "       caption  : download caption data"
        echo "       relation : download predicate, objects data"
        echo ""
        return 1
        ;;
    esac
}
run "$@"


