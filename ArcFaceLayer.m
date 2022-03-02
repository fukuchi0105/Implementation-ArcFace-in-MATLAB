classdef ArcFaceLayer < nnet.layer.Layer & nnet.layer.Formattable% (Optional) 

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        scale
        margin
        cos_m
        sin_m
        th
        mm
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
        W
    end
    
    methods
        function layer = ArcFaceLayer(num_class, scale, margin)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            arguments
                num_class {}
                scale {} = 64.0
                margin {} = 0.50
            end

            layer.scale = scale;
            layer.margin = margin;

            layer.cos_m = cos(margin);
            layer.sin_m = sin(margin);
            layer.th = cos(pi - margin);
            layer.mm = sin(pi - margin) * margin;

            layer.W = randn([512, num_class]);
            layer.W = randn([512, num_class]) * sqrt(1 / (num_class) );%Xavier
            %layer.W = randn([512, num_class]) * sqrt(2 / (num_class) );%%He

            layer.InputNames = {'Input', 'Label'};
            layer.Name = "ArcFaceLayer";

            % Layer constructor function goes here.
        end
        
        function Z = predict(layer, varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Feature
            %         Label       - Ground Truth Label for Penalty
            % Outputs:
            %         Z - Outputs of layer forward function

            X = varargin{1};

            % W size(512, num)
            % X size (512, Batch)
            Xn = X ./ vecnorm(extractdata(X));% normalization

            Wt = layer.W;
            Wn = Wt ./ vecnorm(extractdata(Wt));% normalization

            Xn = reshape(Xn, 512, []);%Xnが(1, 1, num, Batch)のときがあるので、(num, Batch)に変更
            cosine = Wn' * stripdims(Xn); %(num, 512) * (512, Batch) -> (num, Batch) cosがNum個×Batch

            Z = layer.scale * cosine;
            Z = dlarray(Z, "CB");
        end

        function Z = forward(layer, varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X           - Feature varargin{1}
            %         Label       - Ground Truth Label for Penalty varargin{2}
            % Outputs:
            %         Z - Outputs of layer forward function

            X = varargin{1};
%             Label = varargin{2};

            % W size(512, num)
            % X size (512, Batch)
            Xn = X ./ vecnorm(extractdata(X));% normalization

            Wt = layer.W;
            Wn = Wt ./ vecnorm(extractdata(Wt));% normalization

            Xn = reshape(Xn, 512, []);%Xnが(1, 1, num, Batch)のときがあるので、(num, Batch)に変更
            cosine = Wn' * stripdims(Xn); %(num, 512) * (512, Batch) -> (num, Batch) cosがNum個×Batch

            % if Label is inputted, ArcFace margin is calculated
            if (numel(varargin) > 1)
                Label = varargin{2};
                sine = sqrt(1.0 - cosine.^2);%(num, Batch) sinは正なので第1,第2象限に限定　0 <= theta <= pi　つまりtheta==0が最適解
                phi = cosine.*layer.cos_m - sine.*layer.sin_m; %加法定理でcos(θ+m)を計算
    
                %phi < cosineのほうがよいか？　いや、よくない
                phi_valid = cosine > layer.th; %layer.th = cos(π - m) < cos(θ) の状態を判定　ここにmを足すとcosπを通り越して減少に転じる可能性がある
                phi = phi.*phi_valid + ((cosine - layer.mm) .* (~phi_valid));%[mm = sin(pi - margin) * margin] これってlossは下げるけど、微分値としては意味ない気がする
                
                phi_hot = Label .* phi;
                cos_hot = (1.0 - Label) .* cosine;
    
                Z = phi_hot + cos_hot;
            else
                Z = cosine;
            end
            
            Z = layer.scale * Z;
            Z = dlarray(Z, "CB");
            % Layer forward function for training goes here.
        end
        function layer = makeSingleInputCopy(layer)
            layer.InputNames = {'Input'};
        end

%         function [dLdX, dLdLabel, dLdW] = ...
%                 backward(layer, X, Label, Z, dLdZ, memory)
%             % (Optional) Backward propagate the derivative of the loss  
%             % function through the layer.
%             %
%             % Inputs:
%             %         layer             - Layer to backward propagate through
%             %         X1, ..., Xn       - Input data
%             %         Z1, ..., Zm       - Outputs of layer forward function            
%             %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
%             %         memory            - Memory value from forward function
%             % Outputs:
%             %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
%             %                             inputs
%             %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
%             %                             learnable parameter
%      
%             % W size(512, num)
%             % X size (512, Batch)
%             % Z size (num, Batch)
% 
%             dLdX = layer.W * dLdZ; %(512, num) * (num, Batch) -> (512, Batch)
%             dLdLabel = zeros(size(Label));
%             dLdW = X * dLdZ'; %(512, Batch) * (Batch, num) -> (512, num)
%         end

    end
end