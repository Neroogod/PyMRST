"""Dataset download predicates ported from MRST."""


def dataset_has_valid_file_url(info):
    return isinstance(info.fileurl, str) and bool(info.fileurl)


def dataset_has_custom_download_fcn(info):
    return callable(info.downloadFcn)


datasetHasValidFileURL = dataset_has_valid_file_url
datasetHasCustomDownloadFcn = dataset_has_custom_download_fcn

