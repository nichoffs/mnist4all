document.addEventListener("DOMContentLoaded", function() {
    class Function {
        constructor(...tensors) {
            this.parents = tensors;
        }
        static apply(...tensors) {
            let ctx = new this(...tensors);
            let ret = new Tensor(ctx.forward(...tensors.map(t => t.buf)));
            ret._ctx = ctx;
            return ret;
        }
    }

    class Tensor {
        constructor(buf) {
            this.buf = buf;
        }
        shape() {
            return this.buf.shape;
        }
        toString(data = false) {
            return "<Tensor with" + (data ? " data: " + this.buf.toString() : "") + " shape: " + this.shape() + ">";
        }
        mul(other) {
            return Mul.apply(this, other);
        }
        sum() {
            return Sum.apply(this);
        }
        dot(other) {
            return Dot.apply(this, other);
        }
        logsoftmax() {
            return LogSoftmax.apply(this);
        }

    }





});
