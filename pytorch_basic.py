import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch_basic.py!')


def create_sample_tensor():

    x = None

    x = torch.zeros([3, 2])
    x[0, 1] = 10
    x[1, 0] = 100

    return x


def mutate_tensor(x, indices, values):


    for i in range(len(indices)):
        x[indices[i]] = values[i]


    return x


def count_tensor_elements(x):

    num_elements = None

    num_elements = torch.numel(x)

    return num_elements


def create_tensor_of_pi(M, N):

    x = None

    x = 3.14 * torch.ones(M, N)
    return x


def multiples_of_ten(start, stop):

    assert start <= stop
    x = None

    L = []
    for i in range(start, stop + 1):
        if i == 0:
            continue

        if i % 10 == 0:
            L.append(i)
    x = torch.tensor(L, dtype=torch.float64)


    return x


def slice_indexing_practice(x):

    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None

    last_row = x[-1, :]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[0::2, 1::2]

    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x):

    x[:2, 0] = 0
    x[:2, 1] = 1
    x[:2, 2:6] = 2
    x[2:4, 0] = 3
    x[2:4, 1] = 4
    x[2:4, 2] = 3
    x[2:4, 3] = 4
    x[2:4,4:6]=5

    return x


def shuffle_cols(x):

    idx=[0,0,2,1]
    y=x[:,idx]

    return y


def reverse_rows(x):

    l=[x.shape[0] - 1 - i for i in range(x.shape[0])]
    y=x[l,:]

    return y


def take_one_elem_per_col(x):

    y=x[[1,0,3],[0,1,2]]

    return y


def count_negative_entries(x):

    num_neg = 0

    mask = x < 0
    num_neg=x[mask].shape[0]

    return num_neg


def make_one_hot(x):

    y = None

    C=max(x)+1
    N=len(x)
    y=torch.zeros((N,C),dtype=torch.float32)
    rows=[i for i in range(len(x))]
    y[rows,x]=1

    return y


def reshape_practice(x):

    y = None

    y=x.view(2,3,4).permute(1,0,2).reshape(3,8)


    return y


def zero_row_min(x):

    y = None

    min_ind=x.min(dim=1)[1]
    x[torch.arange(len(x)),min_ind]=0
    y=x.clone()

    return y


def batched_matrix_multiply(x, y, use_loop=True):

    z = None

    z = torch.zeros((x.shape[0],x.shape[1], y.shape[2]), dtype=x.dtype)
    if use_loop==True:
        for i in range(len(x)):
            z[i] = torch.matmul(x[i], y[i])
    else:
        z=torch.bmm(x,y)

    return z


def normalize_columns(x):

    y = None

    mu = x.sum(dim=0) / x.shape[0]
    squared_diff = (x - mu) ** 2
    sigma = torch.sqrt(squared_diff.sum(dim=0) / (x.shape[0] - 1))
    y=(x - mu)/sigma

    return y


def mm_on_cpu(x, w):

    y = x.mm(w)
    return y


def mm_on_gpu(x, w):

    y = None

    x_gpu=x.cuda()
    w_gpu=w.cuda()
    torch.cuda.synchronize()
    y2=torch.matmul(x_gpu,w_gpu)
    y=y2.cpu()

    return y
