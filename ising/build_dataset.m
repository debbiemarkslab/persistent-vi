function [X, h, J] = generate_toy_dataset(type, samples)
%GENERATE_TOY_DATASETS generates toy binary Ising systems.
%
if nargin == 0
    type = 'ferrocube';
    samples = 1E4;
end

switch type
    case {'ferrocube'}
        %
        % Generate h, J for a cubic ferromagnet.
        %
        % The interaction topology is a periodic 3D lattice,
        % where J = <pos. constant> for all adjacent nodes
        % and is 0 everywhere else.
        %
        k = 2; % Level of recursion
        J_0 = 0.2; % Coupling strength
        
        m = 2^k;
        [Ix, Jx, Kx] = hilbert3(k);
        Ix = (Ix + 0.5) * 2^k + 0.5;
        Jx = (Jx + 0.5) * 2^k + 0.5;
        Kx = (Kx + 0.5) * 2^k + 0.5;
        
        % Add periodic boundaries by replicating the system in
        % every direction
        X = [Ix(:) Jx(:) Kx(:)];
        D = cat(3, pdist2(X, [Ix(:) Jx(:) Kx(:)],   'cityblock'), ...
            pdist2(X, [Ix(:)+m Jx(:) Kx(:)], 'cityblock'), ...
            pdist2(X, [Ix(:)-m Jx(:) Kx(:)], 'cityblock'), ...
            pdist2(X, [Ix(:) Jx(:)+m Kx(:)], 'cityblock'), ...
            pdist2(X, [Ix(:) Jx(:)-m Kx(:)], 'cityblock'), ...
            pdist2(X, [Ix(:) Jx(:) Kx(:)+m], 'cityblock'), ...
            pdist2(X, [Ix(:) Jx(:) Kx(:)-m], 'cityblock'));
        D = min(D,[],3);
        C = abs(D - 1) < 0.01;
        
        % Ferromagnet with J just below critical
        h = zeros(size(C,1),1);
        J = C * J_0;
    case {'spinglass'}
        %
        % Generate h, J for a (dilute) Sherrington-Kirkpatrick spin glass.
        %
        % The interaction topology is an Erdos-Renyi graph and the 
        % interaction strengths are distributed as N(0, sqrt(1 / (Np))).
        %
        N = 100; % System size
        p = 0.02; % Coupling 
        
        h = zeros(N, 1);
        Nj = N * (N-1) / 2;
        J_flat = (rand(Nj,1) < p) .* randn(Nj,1) * sqrt(1 / (N * p));
        J = squareform(J_flat);
end

% Swendsen-Wang sampling
burn_in = 100;
X = sample_sw(h, J, burn_in, samples);
end

function [x,y,z] = hilbert3(n)
% Hilbert 3D curve.
%
% function [x,y,z] = hilbert3(n) gives the vector coordinates of points
% in n-th order Hilbert curve of area 1.
%
% Example: plot the 3-rd order curve
%
% [x,y,z] = hilbert3(3); plot3(x,y,z)

%   Copyright (c) by Ivan Martynov
%   Inspired by function [x,y] = hilbert(n) made by Federico Forte
%   Date: September 17, 2009

if nargin ~= 1
    n = 2;
end

if n <= 0
    x = 0;
    y = 0;
    z = 0;
else
    [xo,yo,zo] = hilbert3(n-1);
    x = .5*[.5+zo .5+yo -.5+yo -.5-xo -.5-xo -.5-yo .5-yo .5+zo];
    y = .5*[.5+xo .5+zo .5+zo .5+yo -.5+yo -.5-zo -.5-zo -.5-xo];
    z = .5*[.5+yo -.5+xo -.5+xo .5-zo .5-zo -.5+xo -.5+xo .5-yo];
end
end

