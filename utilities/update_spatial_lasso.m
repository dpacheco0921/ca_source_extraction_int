function [A,C] = update_spatial_lasso(Y, A, C, IND, sn, q, maxIter, options)

%% update spatial components using constrained non-negative lasso with warm started HALS 

% input: 
%       Y:    d x T,  fluorescence data
%       A:    d x K,  spatial components + background
%       C:    K x T,  temporal components + background
%     IND:    K x T,  spatial extent for each component
%      sn:    d x 1,  noise std for each pixel
%       q:    scalar, control probability for FDR (default: 0.975)
% maxIter:    maximum HALS iteration (default: 40)
% options:    options structure

% output: 
%   A: d*K, updated spatial components 

% Author: Eftychios A. Pnevmatikakis

%% options for HALS

memmaped = isobject(Y);

%norm_C_flag = false;
tol = 1e-3;
repeat = 1;
defoptions = CNMFSetParms;
if nargin < 8; options = defoptions; end
if nargin < 7 || isempty(maxIter); maxIter = 40; end
if nargin < 6 || isempty(q); q = 0.75; end
if   nargin<5 || isempty(sn); sn = get_noise_fft(Y,options);  end;
if   nargin<4 || isempty(IND); IND = determine_search_location(A,options.search_method,options); end 
if nargin < 2 || isempty(A); 
    A = max(Y*C'/(C*C'),0);
end
if ~isfield(options,'spatial_parallel'); 
    spatial_parallel = ~isempty(which('parpool')); 
else
    spatial_parallel = options.spatial_parallel;
end % use parallel toolbox if present

% if norm_C_flag
%     nC = sqrt(sum(C.^2,2));
%     A = bsxfun(@times,A,nC);
%     C = bsxfun(@times,C,1./nC(:));
% end

[d,K] = size(A);

nr = K - options.nb;
IND(:,nr+1:K) = true;
T = size(C,2);
step_size = 2e4;

if memmaped
    %d = size(A,1);
    sizY = Y.sizY;
    if spatial_parallel
        chunks = 1:step_size:d;
        Yf = cell(numel(chunks),1); %zeros([d,numel(nr+1:K)]);
        parfor t = 1:numel(chunks)
            Yf{t} = double(Y.Yr(chunks(t):min(chunks(t)+step_size-1,d),:))*C(nr+1:end,:)';
        end
        Yf = cell2mat(Yf);
    else
        Yf = zeros([d,numel(nr+1:K)]);
        for t = 1:step_size:d
            Yf(t:min(t+step_size-1,d),:) = double(Y.Yr(t:min(t+step_size-1,d),:))*C(nr+1:end,:)';
        end
    end
    YC = [];
else
    YC = double(Y*C');
    Yf = YC(:,nr+1:end); %Y*f';
end

%% initialization 
if numel(A) > step_size
    for i = 1:K
        A(~IND(:,K),K) = 0;
    end
else
    A(~IND) = 0; 
end
U = YC; 
V = C*C'; 
cc = diag(V);   % squares of l2 norm for all components 

%% updating (neuron by neuron)
miter = 0;
while repeat && miter < maxIter
    A_ = A;
    if spatial_parallel
        Ai = cell(K,1);
        tmp_ind = cell(K,1);
        for k=1:K
            tmp_ind{k,1} = IND(:,k);
            Ai{k,1} = A(tmp_ind{k,1},:);
            if mod(k,300) == 0
                fprintf('%2.1f%% of pixels completed \n', k*100/K);
            end
        end
        fprintf('\n')
        parfor k=1:K
            fprintf([num2str(k), '\n'])
            % local computation of U
            if memmaped
                if k <= nr
                    % Get subvolume
                    sv_idx = getsubvolumeidx(tmp_ind{k,1}, sizY(1:end-1));
                    tmp_ind_i = getsubvolume(tmp_ind{k,1}, sv_idx, sizY(1:end-1));
                    tmp_ind_i = tmp_ind_i{1}; sv_idx = sv_idx{1};
                    if size(sv_idx, 2) == 2
                        Ul = Y.Y(sv_idx(1,1):sv_idx(2,1), sv_idx(1,2):sv_idx(2,2), :);
                    else
                        Ul = Y.Y(sv_idx(1,1):sv_idx(2,1), sv_idx(1,2):sv_idx(2,2), sv_idx(1,3):sv_idx(2,3), :);
                    end
                    Ul = reshape(Ul, numel(tmp_ind_i), T);
                    Ul = double(Ul(tmp_ind_i>0,:)*C(k, :)');
                else
                    Ul = Yf(:, K - nr);
                end
            else
                Ul = U(tmp_ind{k,1}, k);
            end
            
            if k <= nr
                lam = sqrt(cc(k)); %max(sqrt(cc(tmp_ind)));
            else
                lam = 0;
            end
            
            LAM = norminv(q)*sn*lam;
            
            ak{k,1} = max(0, full(Ai{k,1}(:, k))+(Ul - LAM(tmp_ind{k,1}) - full(Ai{k,1}*V(:, k)))/cc(k));
            
            if mod(k,300) == 0
                fprintf('%2.1f%% of pixels completed \n', k*100/K);
            end           
        end
        fprintf('\n')
        for k=1:K
            A(tmp_ind{k,1}, k) = ak{k,1};
            if mod(k,300) == 0
                fprintf('%2.1f%% of pixels completed \n', k*100/K);
            end
        end
        fprintf('\n')
        clear tmp_ind ak
    else
        for k=1:K
            fprintf('*')
            tmp_ind = IND(:, k);
            % local computation of U
            if memmaped
                % Get subvolume
                sv_idx = getsubvolumeidx(tmp_ind, sizY(1:end-1));
                tmp_ind_i = getsubvolume(tmp_ind, sv_idx, sizY(1:end-1));
                tmp_ind_i = tmp_ind_i{1}; sv_idx = sv_idx{1};
                if size(sv_idx, 2) == 2
                    Ul = Y.Y(sv_idx(1,1):sv_idx(2,1), sv_idx(1,2):sv_idx(2,2), :);
                else
                    Ul = Y.Y(sv_idx(1,1):sv_idx(2,1), sv_idx(1,2):sv_idx(2,2), sv_idx(1,3):sv_idx(2,3), :);
                end
                Ul = reshape(Ul, numel(tmp_ind_i), T);
                Ul = double(Ul(tmp_ind_i>0,:)*C(k, :)');
                clear sv_idx tmp_ind_i
            else
                Ul = U(tmp_ind, k);
            end
            if k <= nr
                lam = sqrt(cc(k)); %max(sqrt(cc(tmp_ind)));
            else
                lam = 0;
            end
            LAM = norminv(q)*sn*lam;
            ak = max(0, full(A(tmp_ind, k))+(Ul - LAM(tmp_ind) - full(A(tmp_ind,:)*V(:, k)))/cc(k)); 
            A(tmp_ind, k) = ak;
        end
    end
    miter = miter + 1;
    repeat = (sqrt(sum((A(:)-A_(:)).^2)/sum(A_(:).^2)) > tol);    
end

f = C(nr+1:end,:);
b = double(max((double(Yf) - A(:,1:nr)*double(C(1:nr,:)*f'))/(f*f'),0));
A(:,nr+1:end) = b;