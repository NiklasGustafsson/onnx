// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE;
#ifdef ONNX_ML
ONNX_OPERATOR_SCHEMA(ArrayFeatureExtractor)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Select elements of the input tensor based on the indices passed.
)DOC")
.Input(0, "X", "Data to be selected", "T1")
.Input(1, "Y", "The indices, of which must be at least one and no more than the number of dimensions of 'X.'", "T2")
.Output(0, "Z", "Selected output data as an array", "T1")
.TypeConstraint(
    "T1",
    { "tensor(float)",
    "tensor(double)",
    "tensor(int64)",
    "tensor(int32)",
    "tensor(string)" },
    "")
.TypeConstraint("T2", { "tensor(int64)" }, "");

ONNX_OPERATOR_SCHEMA(Binarizer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Replaces the values of the input tensor by either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
)DOC")
.Input(0, "X", "Data to be binarized", "T")
.Output(0, "Y", "Binarized output data", "T")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr(
    "threshold",
    "Values greater than this are replaced by 1, others by 0.<br>The default threshold is 0.0",
    AttributeProto::FLOAT,
    0.f);

ONNX_OPERATOR_SCHEMA(CastMap)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Converts a map to a tensor.<br><br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br><br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
)DOC")
.Input(0, "X", "The input map that is to be cast to a tensor.", "T1")
.Output(0, "Y", "A tensor representing the same data as the input map, ordered by their keys.", "T2")
.TypeConstraint(
    "T1",
    { "map(int64, string)", "map(int64, float)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(float)", "tensor(int64)" },
    "")
.Attr(
    "cast_to",
    "A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'; the default is 'TO_FLOAT'",
    AttributeProto::STRING,
    std::string("TO_FLOAT"))
.Attr(
    "map_form",
    "Indicates whether to only output as many values as are in the input (dense), or position the input based on using the key of the map as the index of the output (sparse).<br>One of 'DENSE', 'SPARSE', default is 'DENSE'",
    AttributeProto::STRING,
    std::string("DENSE"))
.Attr(
    "max_map",
    "If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor.<br>The default value is 1.",
    AttributeProto::INT,
    static_cast<int64_t>(1));

ONNX_OPERATOR_SCHEMA(CategoryMapper)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Converts strings to integers and vice versa.<br><br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br><br>
    Each operator converts either integers to strings or strings to integers, depending 
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br><br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.
)DOC")
.Input(0, "X", "Input data", "T1")
.Output(
    0,
    "Y",
    "Output data. If strings are input, the output values are integers, and vice versa.",
    "T2")
.TypeConstraint(
    "T1",
    { "tensor(string)", "tensor(int64)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(int64)", "tensor(string)" },
    "")
.Attr(
    "cats_strings",
    "The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence.",
    AttributeProto::STRINGS)
.Attr(
    "cats_int64s",
    "The integers of the map. This sequence must be the same length as the 'cats_strings' sequence.",
    AttributeProto::INTS)
.Attr(
    "default_string",
    "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
    AttributeProto::STRING,
    OPTIONAL)
.Attr(
    "default_int64",
    "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
    AttributeProto::INT,
    OPTIONAL);

ONNX_OPERATOR_SCHEMA(DictVectorizer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Uses an index mapping to convert a dictionary to an array.<br><br>
    Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    the key type. The index into the vocabulary array at which the key is found is then
    used to index into the output single-dimenstional tensor 'Y' and insert the
    the value found in the dictionary 'X'.<br><br>
    The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    Therefore, the output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    Any keys not present in the input dictionary, will be zero in the output array.<br><br>
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    )DOC")
.Input(0, "X", "A dictionary", "T1")
.Output(0, "Y", "A tensor holding values from the input dictionary.", "T2")
.TypeConstraint(
    "T1",
    { "map(string, int64)", "map(int64, string)", "map(int64, float)", "map(int64, double)", "map(string, float)", "map(string, double)"},
    "")
.TypeConstraint(
    "T2",
    { "tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)"},
    "")
.Attr("string_vocabulary", "A string vocabulary array.<br>One and only one of the vocabularies must be defined.", AttributeProto::STRINGS, OPTIONAL)
.Attr("int64_vocabulary", "An integer vocabulary array.<br>One and only one of the vocabularies must be defined.", AttributeProto::INTS, OPTIONAL);

ONNX_OPERATOR_SCHEMA(FeatureVectorizer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Concatenates input features into one continuous output.<br><br>
    **TODO: Correct this explanation, because it refers to an attribute that does not exist:**<br>
    Inputlist is a list of input feature names, inputdimensions is the size of each input feature.
    Inputs will be written to the output in the order of the input arguments.<br><br>
    All inputs are tensors of float. Any feature that is not a tensor of float should
    be converted using either Cast or CastMap.
)DOC")
.Input(0, "X", "An ordered collection of tensors, all with the same element type.", "T1", OpSchema::Variadic)
.Output(0, "Y", "The ", "T2")
.TypeConstraint("T1", { "tensor(int32)", "tensor(int64)", "tensor(float)", "tensor(double)" }, " Allowed input types")
.TypeConstraint("T2", { "tensor(float)" }, " Output data type")
.Attr("inputdimensions", "The size of each input in the input list", AttributeProto::INTS);

ONNX_OPERATOR_SCHEMA(Imputer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Replaces inputs that equal one value with another, leaving all other elements alone.<br><br>
    This operator is typically used to replace missing values in situations where missing values have a canonical
    representation, such as -1, 0, or some extreme value.<br><br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br><br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature. In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
)DOC")
.Input(0, "X", "Data to be processed", "T")
.Output(0, "Y", "Imputed output data", "T")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr("imputed_value_floats", "Value(s) to change to", AttributeProto::FLOATS, OPTIONAL)
.Attr("replaced_value_float", "A value that needs replacing", AttributeProto::FLOAT, 0.f)
.Attr("imputed_value_int64s", "Value(s) to change to", AttributeProto::INTS, OPTIONAL)
.Attr("replaced_value_int64", "A value that needs replacing", AttributeProto::INT, static_cast<int64_t>(0));

ONNX_OPERATOR_SCHEMA(LabelEncoder)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Converts strings to integers and vice versa.<br><br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.<br><br>
    Each operator converts either integers to strings or strings to integers, depending 
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br><br>
    When converting from integers to strings, the string is fetched from the
    'classes_strings' list, by simple indexing.<br><br>
    When converting from strings to integers, the string is looked up in the list
    and the index at which it is found is used as the converted value.
)DOC")
.Input(0, "X", "Input data", "T1")
.Output(
    0,
    "Y",
    "Output data. If strings are input, the output values are integers, and vice versa.",
    "T2")
.TypeConstraint(
    "T1",
    { "tensor(string)", "tensor(int64)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(int64)", "tensor(string)" },
    "")
.Attr(
    "classes_strings",
    "A list of labels",
    AttributeProto::STRINGS)
.Attr(
    "default_int64",
    "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
    AttributeProto::INT,
    OPTIONAL)
.Attr(
    "default_string",
    "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
    AttributeProto::STRING,
    OPTIONAL);

ONNX_OPERATOR_SCHEMA(LinearClassifier)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Linear classifier.
)DOC")
.Input(0, "X", "Data to be classified", "T1")
.Output(0, "Y", "Classification outputs (one class per example", "T2")
.Output(
    1,
    "Z",
    "Classification scores ([N,E] - one score for each class and example)",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    "")
.Attr("coefficients", "A collection of weights of the model(s)", AttributeProto::FLOATS)
.Attr("intercepts", "A collection of intercepts.", AttributeProto::FLOATS)
.Attr(
    "multi_class",
    "Indicates whether to do OvR or multinomial (0=OvR is the default).",
    AttributeProto::INT,
    static_cast<int64_t>(0))
.Attr(
    "classlabels_strings",
    "Class labels when using string labels. One and only one 'classlabels' attribute must be defined.",
    AttributeProto::STRINGS,
    OPTIONAL)
.Attr(
    "classlabels_ints",
    "Class labels when using integer labels. One and only one 'classlabels' attribute must be defined.",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "post_transform",
    "Indicates the transform to apply to the scores vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    std::string("NONE"));

ONNX_OPERATOR_SCHEMA(LinearRegressor)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Generalized linear regression evaluation.<br><br>
    If targets is set to 1 (default) then univariate regression is performed.<br><br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br><br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC")
.Input(0, "X", "Data to be regressed", "T")
.Output(
    0,
    "Y",
    "Regression outputs (one per target, per example)",
    "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr(
    "post_transform",
    "Indicates the transform to apply to the regression output vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    std::string("NONE"))
.Attr("coefficients", "Weights of the model(s)", AttributeProto::FLOATS)
.Attr("intercepts", "Weights of the intercepts, if used", AttributeProto::FLOATS, OPTIONAL)
.Attr(
    "targets",
    "The total number of regression targets, 1 if not defined.",
    AttributeProto::INT,
    static_cast<int64_t>(1));

ONNX_OPERATOR_SCHEMA(Normalizer)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br><br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br><br>
    In all modes, if the divisor is zero, Y == X.
)DOC")
.Input(0, "X", "Data to be encoded", "T")
.Output(0, "Y", "encoded output data", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr("norm", "One of 'MAX,' 'L1,' 'L2'", AttributeProto::STRING, std::string("MAX"));

ONNX_OPERATOR_SCHEMA(OneHotEncoder)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count 
    will determine the sized of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8, 
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br><br>
    This operator assumes every input feature is from the same set of categories.<br><br>
	If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.
)DOC")
.Input(0, "X", "Data to be encoded.", "T")
.Output(0, "Y", "Encoded output data, having one more dimension than X.", "tensor(float)")
.TypeConstraint("T", { "tensor(string)", "tensor(int64)","tensor(int32)", "tensor(float)","tensor(double)" }, "")
.Attr("cats_int64s", "List of categories, ints.<br>One and only one of the 'cats_*' attributes must be defined.", AttributeProto::INTS, OPTIONAL)
.Attr("cats_strings", "List of categories, strings.<br>One and only one of the 'cats_*' attributes must be defined.", AttributeProto::STRINGS, OPTIONAL)
.Attr(
    "zeros",
    "If true and category is not present, will return all zeros; if false and a category if not found, the operator will fail.",
    AttributeProto::INT,
    static_cast<int64_t>(1));

ONNX_OPERATOR_SCHEMA(Scaler)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC")
.Input(0, "X", "Data to be scaled", "T")
.Output(0, "Y", "Scaled output data", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr(
    "offset",
    "First, offset by this. Can be length of features or length 1, in which case it applies to all features",
    AttributeProto::FLOATS)
.Attr(
    "scale",
    "Second, multiply by this, must be same length as offset.<br>Can be length of features or length 1, in which case it applies to all features.",
    AttributeProto::FLOATS);

ONNX_OPERATOR_SCHEMA(SVMClassifier)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Support Vector Machine classifier
)DOC")
.Input(0, "X", "Data to be classified", "T1")
.Output(0, "Y", "Classification outputs (one class per example)", "T2")
.Output(
    1,
    "Z",
    "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class otherwise they are raw scores.",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    "")
.Attr(
    "kernel_type",
    "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID. The default is 'LINEAR.'",
    AttributeProto::STRING,
    std::string("LINEAR"))
.Attr(
    "kernel_params",
    "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr("vectors_per_class", "", AttributeProto::INTS)
.Attr("support_vectors", "", AttributeProto::FLOATS)
.Attr("coefficients", "", AttributeProto::FLOATS)
.Attr("prob_a", "First set of probability coefficients", AttributeProto::FLOATS, OPTIONAL)
.Attr(
    "prob_b",
    "Second set of probability coefficients. This array must be same size as prob_a.<br>If these are provided then output Z are probability estimates, otherwise raw scores.",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr("rho", "", AttributeProto::FLOATS)
.Attr(
    "post_transform",
    "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    std::string("NONE"))
.Attr(
    "classlabels_strings",
    "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
    AttributeProto::STRINGS,
    OPTIONAL)
.Attr(
    "classlabels_ints",
    "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
    AttributeProto::INTS,
    OPTIONAL);

ONNX_OPERATOR_SCHEMA(SVMRegressor)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Support Vector Machine regression prediction and one-class SVM anomaly detection
)DOC")
.Input(0, "X", "Data to be regressed", "T")
.Output(
    0,
    "Y",
    "Regression outputs (one score per target per example)",
    "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr(
    "kernel_type",
    "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID. The default is 'LINEAR.'",
    AttributeProto::STRING,
    std::string("LINEAR"))
.Attr(
    "kernel_params",
    "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr("support_vectors", "Chosen support vectors", AttributeProto::FLOATS, OPTIONAL)
.Attr(
    "one_class",
    "Flag indicating whether the regression is a one-class SVM or not. The default is false.",
    AttributeProto::INT,
    static_cast<int64_t>(0))
.Attr("coefficients", "Support vector coefficients", AttributeProto::FLOATS)
.Attr("n_supports", "The number of support vectors", AttributeProto::INT, static_cast<int64_t>(0))
.Attr(
    "post_transform",
    "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    OPTIONAL)
.Attr("rho", "", AttributeProto::FLOATS);

ONNX_OPERATOR_SCHEMA(TreeEnsembleClassifier)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Tree Ensemble classifier.  Returns the top class for each of N inputs.<br><br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by 
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br><br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br><br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
)DOC")
.Input(0, "X", "Input of shape [N,F]", "T1")
.Output(0, "Y", "N, Top class for each point", "T2")
.Output(
    1,
    "Z",
    "The class score for each class, for each point, a tensor of shape [N,E].",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    "")
.Attr("nodes_treeids", "Tree id for each node", AttributeProto::INTS, OPTIONAL)
.Attr(
    "nodes_nodeids",
    "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
    AttributeProto::INTS,
    OPTIONAL)
.Attr("nodes_featureids", "Feature id for each node", AttributeProto::INTS, OPTIONAL)
.Attr(
    "nodes_values",
    "Thresholds to do the splitting on for each node.",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr(
    "nodes_hitrates", 
    "Popularity of each node, used for performance and may be omitted.", 
    AttributeProto::FLOATS, 
    OPTIONAL)
.Attr(
    "nodes_modes",
    "Defining the node kind, which implies its behavior. <br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
    AttributeProto::STRINGS,
    OPTIONAL)
.Attr(
    "nodes_truenodeids",
    "Child node if expression is true",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "nodes_falsenodeids",
    "Child node if expression is false",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "nodes_missing_value_tracks_true",
    "For each node, decide if the value is missing (NaN) then use true branch.<br>If undefined, the default is 'false' for all nodes.",
    AttributeProto::INTS,
    OPTIONAL)
.Attr("class_treeids", "The id of the tree that this node is in", AttributeProto::INTS, OPTIONAL)
.Attr("class_nodeids", "node id that this weight is for", AttributeProto::INTS, OPTIONAL)
.Attr(
    "class_ids",
    "The index of the class list that each weight is for",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "class_weights",
    "The weight for the class in class_id",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr(
    "classlabels_strings",
    "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
    AttributeProto::STRINGS,
    OPTIONAL)
.Attr(
    "classlabels_int64s",
    "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "post_transform",
    "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    OPTIONAL)
.Attr(
    "base_values",
    "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
    AttributeProto::FLOATS,
    OPTIONAL);

ONNX_OPERATOR_SCHEMA(TreeEnsembleRegressor)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br><br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br><br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br><br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br><br>
    All trees must have their node ids start at 0 and increment by 1.<br><br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC")
.Input(0, "X", "Input of shape [N,F]", "T")
.Output(0, "Y", "N classes", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    "")
.Attr("nodes_treeids", "Tree id for each node", AttributeProto::INTS, OPTIONAL)
.Attr(
    "nodes_nodeids",
    "Node id for each node. Ids ids must restart at zero for each tree and increase sequentially.",
    AttributeProto::INTS,
    OPTIONAL)
.Attr("nodes_featureids", "Feature id for each node", AttributeProto::INTS, OPTIONAL)
.Attr(
    "nodes_values",
    "Thresholds to do the splitting on for each node.",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr(
    "nodes_hitrates", 
    "Popularity of each node, used for performance and may be omitted.", 
    AttributeProto::FLOATS, 
    OPTIONAL)
.Attr(
    "nodes_modes",
    "Defining the node kind, which implies its behavior. <br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
    AttributeProto::STRINGS,
    OPTIONAL)
.Attr(
    "nodes_truenodeids",
    "Child node if expression is true",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "nodes_falsenodeids",
    "Child node if expression is false",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "nodes_missing_value_tracks_true",
    "For each node, decide if the value is missing (NaN) then use true branch. This field can be left unset and will assume false for all nodes",
    AttributeProto::INTS,
    OPTIONAL)
.Attr("target_treeids", "The id of the tree that each node is in", AttributeProto::INTS, OPTIONAL)
.Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL)
.Attr(
    "target_ids",
    "The index of the target that each weight is for",
    AttributeProto::INTS,
    OPTIONAL)
.Attr(
    "target_weights",
    "The weight for each target",
    AttributeProto::FLOATS,
    OPTIONAL)
.Attr("n_targets", "A total number of targets", AttributeProto::INT, OPTIONAL)
.Attr(
    "post_transform",
    "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
    AttributeProto::STRING,
    OPTIONAL)
.Attr(
    "aggregate_function",
    "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
    AttributeProto::STRING,
    OPTIONAL)
.Attr(
    "base_values",
    "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
    AttributeProto::FLOATS,
    OPTIONAL);

ONNX_OPERATOR_SCHEMA(ZipMap)
.SetDomain("ai.onnx.ml")
.SetDoc(R"DOC(
    Creates a map from the input and the attributes.<br><br>
    The values are provides by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br><br>
    Input 0 may have a batch size larger than 1.<br><br>
    Each input in a batch must be the size of the keys specified by the attributes.<br><br>
    The order of the input and attributes determines the key-value mapping.
)DOC")
.Input(0, "X", "The input values", "tensor(float)")
.Output(0, "Z", "The output map", "T")
.TypeConstraint(
    "T",
    { "map(string, float)", "map(int64, float)" },
    "")
.Attr("classlabels_strings", "The keys when using string keys.<br>One and only one of the 'classlabels_*' attributes must be defined.", AttributeProto::STRINGS, OPTIONAL)
.Attr("classlabels_int64s", "The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes must be defined.", AttributeProto::INTS, OPTIONAL);

#endif
