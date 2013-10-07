function Accu = R2LMTL(path,parameters)
%Implementation of R2LMTL
%Input:
%       path        -- loading path of your dataset
%       parameters  -- structure containing all the hyperparamters
%Output:
%       Accu        -- K-nearest Neighbor accurancy result

%load the dataset
load(path);

%Load the parameters
NumMa_K = parameters.NumMa_K;
Lambda = parameters.lambda;
t0 = parameters.t0;
iter = parameters.iter;
epoch = parameters.epoch;
Kneigh = parameters.kneigh;

%Preprocessing
disp(['Propecessing......']);
[M,Ntr] = size(train_data);
[M,Ncr] = size(cross_data);
[M,Nte] = size(test_data);
D = M-1;%Dimension of the traing points

Ltr = train_data(M,:);
Lcr = cross_data(M,:);
Lte = test_data(M,:);

%Transductive Learning we need to combine the train and test data together
X = [train_data(1:D,:),test_data(1:D,:)];

%Construct the similar matrix S
NA = Ntr+Nte;
S = round(rand(NA,NA));
S = triu(S,1);
SS = zeros(Ntr,Ntr);
for i = 1:Ntr
    for j = (i+1):Ntr
        if Ltr(1,i) == Ltr(1,j)
            SS(i,j) = 1;
        else
            SS(i,j) = 0;
        end
    end
end
S(1:Ntr,1:Ntr) = SS;
S = S+S'+eye(NA);

%Now it is time to minimize the problem
%Define the initial weight for each metric and metric vector
g = zeros(NumMa_K,NA);
for k = 1:NumMa_K
    eval(['L',num2str(k),' = rand(D,D);']);
    if k < NumMa_K
        g(k,:) = (1/(NumMa_K-1))*rand(1,NA);
    end
end

g(NumMa_K,:) = ones(1,NA)-sum(g(1:(NumMa_K-1),:),1);
Index = 1;
while(1)
    for k = 1:NumMa_K
        fprintf('Step 1. Proximal method for Metric L%d.\n',k);
        eval(['L',num2str(k),' = StepOne(L',num2str(k),',g(',num2str(k),',:),X,S,Lambda,t0,iter);']);
    end

    %Step two, use majorization function to approximate the group vector g,
    %here distance metric L is fixed
    L_Con = [];
    for k = 1:NumMa_K
        eval(['L_Con = [L_Con,L',num2str(k),'];']);
    end
    g = StepTwo(X,S,NumMa_K,L_Con,g);

    %Step Three, Tranductive Learning
    S = StepThree(train_data,test_data,S,NumMa_K,L_Con,g);

    Index = Index+1;
    if Index >= epoch
        break;
    end
    clc;
end
%Now we use the test data to test the best accurancy we have
clc;
L_Con = [];
for k = 1:NumMa_K
    eval(['L_Con=[L_Con,L',num2str(k),'];']);
end
disp(['***Testing Phase***'])
[label,accurancy] = testknn(train_data,test_data,L_Con,NumMa_K,g,Kneigh);
%save label label;
disp(['***The final result of classification is ',num2str(accurancy*100),';***']);
Accu = accurancy;